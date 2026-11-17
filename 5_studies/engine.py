"""LoRA encoders, CUDA execution, deterministic loaders, and epoch checkpoints."""
import contextlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import (LoraConfig, TaskType, get_peft_model,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def device_for(cpu_smoke=False):
    if cpu_smoke:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run inside a SLURM GPU allocation; "
                           "--cpu-smoke is for lightweight debugging only.")
    return torch.device("cuda:0")


def amp_dtype(device, precision="auto"):
    if device.type == "cpu":
        return None
    if precision == "float32":
        return None
    if precision == "bfloat16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but GPU does not support it")
        return torch.bfloat16
    return torch.float16


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


def attach_lora(base, spec, config):
    targets = spec["targets"]
    matched = [name for name, module in base.named_modules()
               if any(name.endswith("." + target) for target in targets)
               and isinstance(module, torch.nn.Linear)]
    expected = base.config.num_hidden_layers * spec["modules_per_layer"]
    if len(matched) != expected:
        raise ValueError(f"LoRA targets: expected {expected} Linear layers, found "
                         f"{len(matched)}: {matched}")
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=targets, bias="none"))
    attached = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if len(attached) != expected:
        raise ValueError(f"Adapter attachment mismatch: {attached}")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise ValueError("Expected only LoRA parameters to be trainable")
    info = dict(target_layers=matched, attached_layers=attached,
                trainable_parameters=sum(p.numel() for _, p in trainable),
                total_parameters=sum(p.numel() for p in model.parameters()))
    return model, info


def load_encoder(spec, config, device):
    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], revision=spec.get("revision", "main"))
    # Eager attention avoids optional FlashAttention dependencies and unpadding differences.
    kwargs = {"attn_implementation": "eager"}
    if spec["targets"] != ["Wqkv"]:
        kwargs["add_pooling_layer"] = False
    base = AutoModel.from_pretrained(spec["checkpoint"], revision=spec.get("revision", "main"), **kwargs)
    if getattr(base.config, "model_type", "") == "modernbert":
        base.config.reference_compile = False
    revision = getattr(base.config, "_commit_hash", None)
    model, info = attach_lora(base, spec, config)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)
    info["resolved_revision"] = revision
    return model, tokenizer, info


def encode(model, tokens):
    hidden = model(**tokens).last_hidden_state.float()
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(pooled, p=2, dim=-1)


class PairDataset(Dataset):
    def __init__(self, frame, papers):
        self.rows = frame[["bibcode_a", "bibcode_b", "label"]].to_records(index=False)
        self.texts = (papers.set_index("bibcode").title + "\n\n" +
                      papers.set_index("bibcode").abstract).to_dict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, b, label = self.rows[i]
        return self.texts[a], self.texts[b], int(label)


@dataclass
class Collator:
    tokenizer: object
    max_length: int

    def __call__(self, batch):
        tokens = self.tokenizer([x[0] for x in batch] + [x[1] for x in batch],
                                padding=True, truncation=True, max_length=self.max_length,
                                return_tensors="pt")
        return tokens, torch.tensor([x[2]*2-1 for x in batch], dtype=torch.float32)


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(frame, papers, tokenizer, config, device, shuffle=False, seed=67):
    return DataLoader(PairDataset(frame, papers), batch_size=config["microbatch_size"],
                      shuffle=shuffle, collate_fn=Collator(tokenizer, config["max_length"]),
                      num_workers=config["workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(seed))


def move_batch(tokens, labels, device):
    return ({k: v.to(device, non_blocking=True) for k, v in tokens.items()},
            labels.to(device, non_blocking=True))


@torch.no_grad()
def score(model, loader, device, dtype, margin):
    model.eval()
    losses, scores, labels_out = 0., [], []
    for tokens, labels in loader:
        tokens, labels = move_batch(tokens, labels, device)
        with autocast(device, dtype):
            left, right = encode(model, tokens).chunk(2)
        loss = F.cosine_embedding_loss(left, right, labels, margin=margin)
        losses += loss.item() * len(labels)
        scores.extend((left * right).sum(-1).cpu().tolist())
        labels_out.extend((labels > 0).int().cpu().tolist())
    return losses / len(loader.dataset), np.asarray(labels_out), np.asarray(scores)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, dtype, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    accumulation = config["batch_size"] // config["microbatch_size"]
    saw_gradient = False
    for step, (tokens, labels) in enumerate(loader):
        tokens, labels = move_batch(tokens, labels, device)
        with autocast(device, dtype):
            left, right = encode(model, tokens).chunk(2)
        loss = F.cosine_embedding_loss(left, right, labels, margin=config["margin"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; check AMP precision and input data")
        # Weight by actual pairs in this optimizer window, including the final partial one.
        window_start = (step // accumulation) * accumulation * config["microbatch_size"]
        window_pairs = min(config["batch_size"], len(loader.dataset)-window_start)
        scaler.scale(loss * len(labels) / window_pairs).backward()
        if not saw_gradient:
            grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
            if not grads or not any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
                raise RuntimeError("No finite, nonzero LoRA gradients")
            if next(model.parameters()).device != device or tokens["input_ids"].device != device:
                raise RuntimeError("Model/input device mismatch")
            saw_gradient = True
        if (step+1) % accumulation == 0 or step+1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def restore_adapter(model, state):
    set_peft_model_state_dict(model, state)
    actual = adapter_state(model)
    if set(actual) != set(state) or any(not torch.equal(actual[k], state[k].cpu()) for k in actual):
        raise RuntimeError("Adapter checkpoint did not restore exactly")


def rng_state(loader):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if loader.pin_memory else [],
                loader=loader.generator.get_state())


def restore_rng(state, loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load_checkpoint(path):
    # Trusted, locally generated checkpoints only (includes RNG Python objects).
    return torch.load(path, map_location="cpu", weights_only=False)


def runtime_info(device):
    import peft
    import transformers
    info = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                cuda_version=torch.version.cuda, device=str(device),
                slurm={k: os.environ.get(k) for k in
                       ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                        "CUDA_VISIBLE_DEVICES")})
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=props.name, gpu_total_memory_bytes=props.total_memory)
    return info


class Selection:
    def __init__(self, patience):
        self.patience = patience
        self.best_auc = -1.
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, auc, epoch):
        improved = auc > self.best_auc
        if improved:
            self.best_auc, self.best_epoch, self.bad_epochs = auc, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience
