import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

ROOT = Path.cwd().resolve()
if ROOT.name == "3_train":
    ROOT = ROOT.parent
DATA_PATH = Path(
    "/mnt/research/deepthought/SciAR/SciAR/data/sciar_refereed_v1/papers.parquet"
)
READER_PATH = ROOT / "2_Data/data_reader.ipynb"
OUTPUT_DIR = ROOT / f"5_experiments/outputs/{sys.argv[2]}"
PAIR_READER = "specter"  # 'specter' or 'random'
SPECTER_MODEL = "allenai/specter"
SPECTER_CACHE_DIR = ROOT / "data/cache/specter"
SPECTER_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
SPECTER_PRECISION = "float16" if torch.cuda.is_available() else "float32"
SPECTER_BATCH_SIZE = 32
SPECTER_MATCH_BATCH_SIZE = 16
SPECTER_PAIR_METHOD = "matched_distinct_v1"
SPECTER_PAPER_LIMIT = 2
SPECTER_AUTHOR_LIMIT = 10

MODEL_NAME = sys.argv[1]
print(MODEL_NAME)
SEED = int(sys.argv[3])
MAX_PAPERS = None  # Set a number for a smaller trial.
TRAIN_PAIRS = 250_000
VAL_PAIRS = 10_000
TEST_PAIRS = 10_000
MAX_LENGTH = 512  # Per paper, including title and abstract.
BATCH_SIZE = 8  # Each pair contains two independently encoded papers.
EPOCHS = 7
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MARGIN = 0.2
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
USE_AMP = USE_CUDA
assert DATA_PATH.is_file(), f"Dataset not found: {DATA_PATH}"
assert READER_PATH.is_file(), f"Reader not found: {READER_PATH}"
assert PAIR_READER in ("random", "specter")
assert all(n >= 2 and n % 2 == 0 for n in (TRAIN_PAIRS, VAL_PAIRS, TEST_PAIRS))
assert EPOCHS >= 1 and BATCH_SIZE >= 1 and 8 <= MAX_LENGTH <= 512
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
print(f"Device: {DEVICE} | Mixed precision: {USE_AMP} | Output: {OUTPUT_DIR}")


reader_cells = json.loads(READER_PATH.read_text())["cells"]
reader = {}
for cell in reader_cells:
    if cell["cell_type"] == "code" and "reader-code" in cell.get("metadata", {}).get(
        "tags", []
    ):
        exec("".join(cell["source"]), reader)  # noqa: S102
load_papers = reader["load_papers"]
split_authors = reader["split_authors"]
make_pairs = reader["make_pairs"]

all_papers = load_papers(DATA_PATH, max_papers=MAX_PAPERS)
papers = all_papers
if PAIR_READER == "specter":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "specter_reader", ROOT / "2_Data/specter_reader.py"
    )
    specter_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(specter_reader)
    retained_indices, dedup_stats = specter_reader.deduplicate_works(all_papers)
    papers = all_papers.iloc[retained_indices].reset_index(drop=True)
    (OUTPUT_DIR / "dedup_stats.json").write_text(json.dumps(dedup_stats, indent=2))
    print("Duplicate-work filter:", dedup_stats)
assert papers["bibcode"].is_unique, "Expected one row per bibcode."
author_splits = split_authors(papers, seed=SEED)
assert all(len(ids) >= 2 for ids in author_splits.values()), (
    "Need at least two authors per split."
)
assert not (
    set(author_splits["train"]) & set(author_splits["validation"])
    or set(author_splits["train"]) & set(author_splits["test"])
    or set(author_splits["validation"]) & set(author_splits["test"])
)
paper_groups = papers.groupby("first_author_orcid").indices
(OUTPUT_DIR / "author_splits.json").write_text(json.dumps(author_splits, indent=2))
for name, ids in author_splits.items():
    print(
        f"{name}: {len(ids):,} authors, {sum(len(paper_groups[a]) for a in ids):,} papers"
    )


if PAIR_READER == "specter":
    specter_vectors = specter_reader.specter_embeddings(
        all_papers,
        SPECTER_CACHE_DIR,
        model_name=SPECTER_MODEL,
        batch_size=SPECTER_BATCH_SIZE,
        devices=SPECTER_DEVICES,
        precision=SPECTER_PRECISION,
    )
    specter_vectors = specter_vectors[retained_indices]
pair_sets = {}
for offset, (name, size) in enumerate(
    [("train", TRAIN_PAIRS), ("validation", VAL_PAIRS), ("test", TEST_PAIRS)]
):
    if PAIR_READER == "specter":
        pair_sets[name] = specter_reader.make_matched_pairs(
            papers,
            specter_vectors,
            author_splits[name],
            size,
            paper_limit=SPECTER_PAPER_LIMIT,
            author_limit=SPECTER_AUTHOR_LIMIT,
            batch_size=SPECTER_MATCH_BATCH_SIZE,
            device=DEVICE,
            seed=SEED + offset,
        )
    else:
        pair_sets[name] = make_pairs(
            papers, paper_groups, author_splits[name], size, SEED + offset
        )
for name, examples in pair_sets.items():
    assert examples["label"].value_counts().to_dict() == {
        0: len(examples) // 2,
        1: len(examples) // 2,
    }
    columns = ["bibcode_a", "bibcode_b", "label"]
    if "specter_similarity" in examples:
        columns.append("specter_similarity")
    if "match_id" in examples:
        columns.append("match_id")
    examples[columns].to_csv(OUTPUT_DIR / f"{name}_pairs.csv", index=False)
    print(f"{name}: {len(examples):,} pairs")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)
model = get_peft_model(
    base_model,
    LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
    ),
).to(DEVICE)
model.print_trainable_parameters()


def collate_pairs(batch):
    # first half is A, second half is B; encode each paper separately
    texts = [row["text_a"] for row in batch] + [row["text_b"] for row in batch]
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    # CosineEmbeddingLoss expects -1 for different authors, while saved labels stay 0/1
    labels = torch.tensor(
        [1 if row["label"] == 1 else -1 for row in batch], dtype=torch.float32
    )
    return tokens, labels


loaders = {
    name: DataLoader(
        examples.to_dict("records"),
        batch_size=BATCH_SIZE,
        shuffle=(name == "train"),
        collate_fn=collate_pairs,
        num_workers=0,
        pin_memory=USE_CUDA,
        generator=torch.Generator().manual_seed(SEED),
    )
    for name, examples in pair_sets.items()
}


def encode(tokens):
    tokens = {
        key: value.to(DEVICE, non_blocking=USE_CUDA) for key, value in tokens.items()
    }
    hidden = model(**tokens).last_hidden_state
    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(pooled, p=2, dim=-1)


criterion = torch.nn.CosineEmbeddingLoss(margin=MARGIN)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, all_labels, all_scores = 0.0, [], []
    for tokens, labels in tqdm(loader, desc="Evaluate", leave=False):
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            left, right = encode(tokens).chunk(2)
        loss = criterion(
            left.float(), right.float(), labels.to(DEVICE, non_blocking=USE_CUDA)
        )
        total_loss += loss.item() * len(labels)
        all_labels.extend((labels > 0).int().tolist())
        all_scores.extend((left.float() * right.float()).sum(dim=-1).cpu().tolist())
    return total_loss / len(loader.dataset), np.array(all_labels), np.array(all_scores)


baseline_loss, baseline_labels, baseline_scores = evaluate(loaders["validation"])
baseline_auc = float(roc_auc_score(baseline_labels, baseline_scores))
_, baseline_test_labels, baseline_test_scores = evaluate(loaders["test"])
print(f"Before training: validation loss={baseline_loss:.4f}, AUC={baseline_auc:.4f}")
history, best_auc, best_state = [], -float("inf"), None
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for tokens, labels in tqdm(loaders["train"], desc=f"Epoch {epoch}/{EPOCHS}"):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            left, right = encode(tokens).chunk(2)
        loss = criterion(
            left.float(), right.float(), labels.to(DEVICE, non_blocking=USE_CUDA)
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
    val_loss, val_labels, val_scores = evaluate(loaders["validation"])
    val_auc = float(roc_auc_score(val_labels, val_scores))
    row = dict(  # noqa: C408
        epoch=epoch,
        train_loss=total_loss / TRAIN_PAIRS,
        val_loss=val_loss,
        val_auc=val_auc,
    )
    history.append(row)
    print(row)
    if val_auc > best_auc:
        best_auc, best_epoch = val_auc, epoch
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model).items()
        }
set_peft_model_state_dict(model, best_state)
pd.DataFrame(history)


val_loss, val_labels, val_scores = evaluate(loaders["validation"])
fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
finite = np.isfinite(thresholds)
threshold = float(thresholds[finite][np.argmax((tpr - fpr)[finite])])
test_loss, test_labels, test_scores = evaluate(loaders["test"])
metrics = {
    "baseline_val_auc": baseline_auc,
    "best_epoch": best_epoch,
    "baseline_test_auc": float(
        roc_auc_score(baseline_test_labels, baseline_test_scores)
    ),
    "val_auc": float(roc_auc_score(val_labels, val_scores)),
    "val_accuracy": float(accuracy_score(val_labels, val_scores >= threshold)),
    "threshold": threshold,
    "test_loss": test_loss,
    "test_auc": float(roc_auc_score(test_labels, test_scores)),
    "test_accuracy": float(accuracy_score(test_labels, test_scores >= threshold)),
}
model.save_pretrained(OUTPUT_DIR / "adapter")
tokenizer.save_pretrained(OUTPUT_DIR / "adapter")
config = {
    key: str(value) if isinstance(value, Path) else value
    for key, value in list(globals().items())
    if key.isupper() and isinstance(value, (str, int, float, Path))
}
config.update(
    pooling="attention-mask mean, including special tokens", normalize_embeddings=True
)
config["SPECTER_DEVICES"] = SPECTER_DEVICES
(OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)
print(json.dumps(metrics, indent=2))
print(f"Saved to {OUTPUT_DIR}")
