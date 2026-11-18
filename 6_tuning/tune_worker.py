"""One-GPU Optuna worker for the SciBERT quick tuning study.

First run this once to create the fixed pair manifests:
    python 6_tuning/tune_worker.py --prepare

Then run one worker per GPU; each worker adds trials to the same study:
    CUDA_VISIBLE_DEVICES=0 python 6_tuning/tune_worker.py --trials 7
"""

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    "/mnt/research/deepthought/SciAR/SciAR/data/sciar_refereed_v1/papers.parquet"
)
READER_PATH = ROOT / "2_Data/data_reader.ipynb"
SPECTER_READER_PATH = ROOT / "2_Data/specter_reader.py"
OUTPUT_DIR = ROOT / "6_tuning/outputs/scibert_quick"

MODEL_NAME = "allenai/scibert_scivocab_uncased"
SEED = 67
TRAIN_PAIRS = 20_000
VAL_PAIRS = 2_000
EPOCHS = 3
MAX_LENGTH = 512
SPECTER_MODEL = "allenai/specter"
SPECTER_CACHE_DIR = ROOT / "data/cache/specter"
SPECTER_BATCH_SIZE = 32
SPECTER_MATCH_BATCH_SIZE = 16
SPECTER_PAPER_LIMIT = 2
SPECTER_AUTHOR_LIMIT = 10


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create fixed train/validation pair CSVs and exit.",
    )
    parser.add_argument(
        "--trials", type=int, default=7, help="Trials for this worker to add."
    )
    return parser.parse_args()


def reader_functions():
    namespace = {}
    for cell in json.loads(READER_PATH.read_text())["cells"]:
        if cell["cell_type"] == "code" and "reader-code" in cell.get(
            "metadata", {}
        ).get("tags", []):
            exec("".join(cell["source"]), namespace)
    return namespace["load_papers"], namespace["split_authors"]


def prepare_pairs(device):
    """Use the exact training split/sampler, without creating test pairs."""
    output_pairs = {
        name: OUTPUT_DIR / f"{name}_pairs.csv" for name in ("train", "validation")
    }
    if all(path.exists() for path in output_pairs.values()):
        print("Using existing fixed train/validation pair manifests.")
        return

    load_papers, split_authors = reader_functions()
    all_papers = load_papers(DATA_PATH)
    spec = importlib.util.spec_from_file_location("specter_reader", SPECTER_READER_PATH)
    specter_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(specter_reader)
    retained_indices, dedup_stats = specter_reader.deduplicate_works(all_papers)
    papers = all_papers.iloc[retained_indices].reset_index(drop=True)
    author_splits = split_authors(papers, seed=SEED)
    vectors = specter_reader.specter_embeddings(
        all_papers,
        SPECTER_CACHE_DIR,
        model_name=SPECTER_MODEL,
        batch_size=SPECTER_BATCH_SIZE,
        devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"],
        precision="float16" if device.type == "cuda" else "float32",
    )[retained_indices]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "author_splits.json").write_text(json.dumps(author_splits, indent=2))
    (OUTPUT_DIR / "dedup_stats.json").write_text(json.dumps(dedup_stats, indent=2))
    for offset, (name, size) in enumerate(
        (("train", TRAIN_PAIRS), ("validation", VAL_PAIRS))
    ):
        pairs = specter_reader.make_matched_pairs(
            papers,
            vectors,
            author_splits[name],
            size,
            seed=SEED + offset,
            paper_limit=SPECTER_PAPER_LIMIT,
            author_limit=SPECTER_AUTHOR_LIMIT,
            batch_size=SPECTER_MATCH_BATCH_SIZE,
            device=device,
        )
        pairs.to_csv(output_pairs[name], index=False)
        print(f"Saved {name}: {len(pairs):,} pairs")


def load_pairs_with_text():
    pair_sets = {
        name: pd.read_csv(OUTPUT_DIR / f"{name}_pairs.csv")
        for name in ("train", "validation")
    }
    needed = set().union(
        *(set(p.bibcode_a) | set(p.bibcode_b) for p in pair_sets.values())
    )
    papers = pd.read_parquet(DATA_PATH, columns=["bibcode", "title", "abstract"])
    papers = papers[papers.bibcode.isin(needed)]
    if papers.bibcode.duplicated().any() or len(papers) != len(needed):
        raise ValueError(
            "Pair papers are missing or have duplicate bibcodes in the dataset."
        )
    texts = (
        papers.set_index("bibcode").title
        + "\n\n"
        + papers.set_index("bibcode").abstract
    ).to_dict()
    for pairs in pair_sets.values():
        pairs["text_a"] = pairs.bibcode_a.map(texts)
        pairs["text_b"] = pairs.bibcode_b.map(texts)
    return pair_sets


def main():
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if args.prepare:
        prepare_pairs(device)
        return
    if not all(
        (OUTPUT_DIR / f"{name}_pairs.csv").exists() for name in ("train", "validation")
    ):
        raise FileNotFoundError(
            "Run this once first: python 6_tuning/tune_worker.py --prepare"
        )

    pair_sets = load_pairs_with_text()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate(batch):
        texts = [row["text_a"] for row in batch] + [row["text_b"] for row in batch]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [1 if row["label"] else -1 for row in batch], dtype=torch.float32
        )
        return tokens, labels

    def loaders():
        return {
            name: DataLoader(
                pairs.to_dict("records"),
                batch_size=8,
                shuffle=(name == "train"),
                collate_fn=collate,
                num_workers=0,
                pin_memory=use_amp,
                generator=torch.Generator().manual_seed(SEED),
            )
            for name, pairs in pair_sets.items()
        }

    def objective(trial):
        set_seed(SEED)
        rank = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
        alpha_multiplier = trial.suggest_categorical("alpha_multiplier", [1, 2, 4])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        margin = trial.suggest_float("margin", 0.0, 0.5)
        dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
        model = get_peft_model(
            AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False),
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=rank * alpha_multiplier,
                lora_dropout=dropout,
                target_modules=["query", "value"],
                bias="none",
            ),
        ).to(device)
        criterion = torch.nn.CosineEmbeddingLoss(margin=margin)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        def encode(tokens):
            tokens = {
                key: value.to(device, non_blocking=use_amp)
                for key, value in tokens.items()
            }
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return F.normalize(pooled, p=2, dim=-1)

        def validation_auc(loader):
            model.eval()
            labels, scores = [], []
            with torch.no_grad():
                for tokens, targets in tqdm(loader, desc="Evaluate", leave=False):
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_amp
                    ):
                        left, right = encode(tokens).chunk(2)
                    labels.extend((targets > 0).int().tolist())
                    scores.extend((left.float() * right.float()).sum(-1).cpu().tolist())
            return float(roc_auc_score(labels, scores))

        best_auc = -float("inf")
        try:
            for epoch in range(EPOCHS):
                model.train()
                for tokens, targets in tqdm(
                    loaders()["train"],
                    desc=f"Trial {trial.number}, epoch {epoch + 1}/{EPOCHS}",
                ):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_amp
                    ):
                        left, right = encode(tokens).chunk(2)
                    loss = criterion(
                        left.float(),
                        right.float(),
                        targets.to(device, non_blocking=use_amp),
                    )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                best_auc = max(best_auc, validation_auc(loaders()["validation"]))
                trial.report(best_auc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return best_auc
        finally:
            del model
            gc.collect()
            if use_amp:
                torch.cuda.empty_cache()

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(OUTPUT_DIR / 'study.db').resolve()}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    study = optuna.create_study(
        study_name="scibert_quick",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    summary = {
        "best_validation_auc": study.best_value,
        "best_params": study.best_params,
        "completed_trials": len(study.trials),
        "note": "No test pairs were loaded or scored.",
    }
    (OUTPUT_DIR / "best_trial.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
