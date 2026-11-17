"""One independent model/seed run. Never constructs pairs or selects on test scores."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import set_seed

from common import (HERE, configuration, load_dataset, read_json, select_experiment, sha256, write_json)
from engine import (Selection, adapter_state, amp_dtype, device_for, load_checkpoint,
                    load_encoder, make_loader, restore_adapter, restore_rng, rng_state,
                    runtime_info, save_checkpoint, score, train_epoch)
from evaluate import export_scores, plots


def main():
    models, defaults = configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=models)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--data", type=Path, default=HERE / "data/fixed_100k")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.task_id is not None:
        if args.model is not None or args.seed is not None:
            parser.error("Use task ID or model/seed, not both")
        args.model, args.seed = select_experiment(args.task_id)
    if args.model is None or args.seed is None:
        parser.error("Supply --task-id or both --model and --seed")
    config = defaults | (read_json(args.config) if args.config else {})
    for key in ("microbatch_size", "workers"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.gradient_checkpointing:
        config["gradient_checkpointing"] = True
    if (config["microbatch_size"] < 1 or config["batch_size"] < 1 or
            config["batch_size"] % config["microbatch_size"] or
            config["epochs"] < 1 or config["patience"] < 1 or config["workers"] < 0):
        parser.error("Positive batch sizes must divide exactly; epochs/patience >=1, workers >=0")
    if config["precision"] not in ("auto", "float32", "float16", "bfloat16"):
        parser.error("precision must be auto, float32, float16 or bfloat16")
    config["gradient_accumulation_steps"] = config["batch_size"] // config["microbatch_size"]
    device = device_for(args.cpu_smoke)
    pairs, papers, manifest = load_dataset(args.data)
    if args.cpu_smoke and (max(len(f) for f in pairs.values()) > 256 or config["epochs"] > 2):
        parser.error("CPU smoke restricted to <=256 pairs/split and <=2 epochs")
    if not args.cpu_smoke and manifest["sizes"] != config["sizes"]:
        parser.error("Dataset sizes differ from training config; use an explicit study config")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dtype = amp_dtype(device, config["precision"])
    output = args.output_root / args.model / f"seed_{args.seed}"
    run_config = dict(model=args.model, seed=args.seed, model_spec=models[args.model],
                      training=config, dataset_fingerprint=manifest["fingerprint"],
                      dataset=str(args.data.resolve()), cpu_smoke=args.cpu_smoke,
                      code_hashes={f: sha256(HERE / f) for f in
                                   ("train.py", "engine.py", "common.py", "evaluate.py")},
                      precision=str(dtype), pooling="masked mean incl. special tokens; float32; L2")
    if args.resume:
        if read_json(output / "config.json") != run_config:
            raise ValueError("Resume config/dataset differs from saved run")
        if (output / "metrics.json").exists():
            print(f"Run already complete: {output}")
            return
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", run_config)
    model, tokenizer, parameters = load_encoder(models[args.model], config, device)
    if args.resume and (output / "parameters.json").exists():
        previous = read_json(output / "parameters.json")
        if previous["resolved_revision"] != parameters["resolved_revision"]:
            raise ValueError("Pretrained checkpoint revision changed; pin model revision")
    write_json(output / "parameters.json", parameters)
    print(parameters, flush=True)
    loaders = {s: make_loader(f, papers, tokenizer, config, device, s == "train", args.seed)
               for s, f in pairs.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if config["scheduler"] != "constant":
        raise ValueError("Only the constant learning-rate scheduler is supported")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    selection = Selection(config["patience"])
    history, best, elapsed, epoch_start = [], None, 0., 1
    sessions = []
    if args.resume and (output / "last.pt").exists():
        state = load_checkpoint(output / "last.pt")
        restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        selection.__dict__.update(state["selection"])
        history, best, elapsed = state["history"], state["best"], state["elapsed"]
        sessions = state.get("sessions", [])
        epoch_start = state["epoch"] + 1
        restore_rng(state["rng"], loaders["train"])
    sessions.append(runtime_info(device))
    started = time.monotonic()
    # Baseline is a disabled-adapter pretrained encoder, with its own validation threshold.
    if not (output / "baseline_metrics.json").exists():
        if args.resume and epoch_start > 1:
            raise ValueError("Missing baseline exports from interrupted run")
        with model.disable_adapter():
            baseline = {s: score(model, loaders[s], device, dtype, config["margin"])
                        for s in ("validation", "test")}
        export_scores(output, "baseline", pairs, baseline)
    for epoch in range(epoch_start, config["epochs"]+1):
        if selection.stopped:
            break
        tick = time.monotonic()
        train_loss = train_epoch(model, loaders["train"], optimizer, scheduler, scaler,
                                 device, dtype, config)
        val_loss, labels, scores = score(model, loaders["validation"], device, dtype, config["margin"])
        auc = float(roc_auc_score(labels, scores))
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_auc=auc,
                   seconds=time.monotonic()-tick,
                   peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        history.append(row)
        if selection.update(auc, epoch):
            best = adapter_state(model)
            save_checkpoint(output / "best.pt", dict(adapter=best, epoch=epoch, val_auc=auc))
        state = dict(adapter=adapter_state(model), best=best, optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     selection=selection.__dict__, history=history, epoch=epoch,
                     rng=rng_state(loaders["train"]), elapsed=elapsed+time.monotonic()-started,
                     sessions=sessions)
        save_checkpoint(output / "last.pt", state)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(row, flush=True)
    if best is None:
        raise RuntimeError("No best checkpoint available")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    restore_adapter(model, best)
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    evaluations = {s: score(model, loaders[s], device, dtype, config["margin"])
                   for s in ("validation", "test")}
    final = export_scores(output, "finetuned", pairs, evaluations)
    plots(output)
    summary = dict(model=args.model, seed=args.seed, dataset_fingerprint=manifest["fingerprint"],
                   best_epoch=selection.best_epoch, best_validation_auc=selection.best_auc,
                   baseline=read_json(output / "baseline_metrics.json"), finetuned=final,
                   training_seconds=sum(r["seconds"] for r in history),
                   total_active_seconds=elapsed+time.monotonic()-started,
                   average_epoch_seconds=float(np.mean([r["seconds"] for r in history])),
                   peak_memory_bytes=max([r["peak_memory_bytes"] for r in history] +
                                         [torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0]),
                   sessions=sessions, parameters=parameters)
    # Completion marker is written last, after all exports succeed.
    write_json(output / "metrics.json", summary)
    print(f"Completed: {output}")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("CUDA out of memory. Use a smaller microbatch and matching accumulation. "
                         "To change the saved configuration, start a new output root; "
                         "unchanged configurations can resume from the last completed epoch. "
                         f"Use gradient checkpointing if needed. Details: {exc}") from exc
