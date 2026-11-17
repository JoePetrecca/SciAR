"""Explicit preflight: pretrained downloads, LoRA backward/accumulation and checkpoint restore."""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from common import (HERE, canonical_hash, configuration, read_json, sha256, write_json)
from engine import (adapter_state, amp_dtype, autocast, device_for, encode, load_checkpoint,
                    load_encoder, make_loader, restore_adapter, runtime_info,
                    save_checkpoint, train_epoch)


def implementation_hash():
    return canonical_hash({p.name: sha256(p) for p in HERE.glob("*.py")})


def check_report(path):
    report = read_json(path)
    models, config = configuration()
    if (not report.get("passed") or report.get("device") != "cuda:0" or
            not report.get("allocation") or
            report["configuration_hash"] != canonical_hash([models, config]) or
            report["implementation_hash"] != implementation_hash() or
            set(report["models"]) != set(models)):
        raise RuntimeError("Missing/stale GPU verification. Run verify.py inside a GPU allocation.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--report", type=Path, default=HERE / "verification/gpu.json")
    parser.add_argument("--check-report", action="store_true")
    args = parser.parse_args()
    if args.check_report:
        check_report(args.report)
        print("GPU preflight report accepted")
        return
    if not args.cpu and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU preflight requires a SLURM allocation; no jobs are submitted here")
    device = device_for(args.cpu)
    models, config = configuration()
    set_seed(67)
    torch.set_num_threads(2)
    smoke = config | dict(microbatch_size=1, batch_size=2, max_length=32, workers=0)
    papers = pd.DataFrame(dict(bibcode=["a", "b", "c"],
                              title=["Stellar populations", "Galactic evolution", "Protein folding"],
                              abstract=["We study stars and spectra.", "We measure galaxy masses.",
                                        "We analyze molecular structures."]))
    pairs = pd.DataFrame(dict(bibcode_a=["a", "a", "b", "c"],
                              bibcode_b=["b", "c", "a", "b"], label=[1, 0, 1, 0]))
    report = dict(passed=False, device=str(device), allocation=os.environ.get("SLURM_JOB_ID"),
                  configuration_hash=canonical_hash([models, config]),
                  implementation_hash=implementation_hash(), runtime=runtime_info(device), models={})
    for name, spec in models.items():
        model, tokenizer, info = load_encoder(spec, smoke, device)
        dtype = amp_dtype(device)
        tokens = tokenizer(["A scientific paper.", "Another paper."], padding=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        model.eval()
        with torch.no_grad(), autocast(device, dtype):
            embeddings = encode(model, tokens)
        if embeddings.shape != (2, model.config.hidden_size):
            raise AssertionError("Unexpected embedding shape")
        torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2, device=device), atol=1e-5, rtol=1e-5)
        loader = make_loader(pairs, papers, tokenizer, smoke, device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        before = adapter_state(model)
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, device, dtype, smoke)
        after = adapter_state(model)
        if not np.isfinite(loss) or not any(not torch.equal(before[k], after[k]) for k in before):
            raise AssertionError("LoRA optimizer did not update adapters")
        with tempfile.TemporaryDirectory(prefix="sciar-verify-") as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, dict(adapter=after, optimizer=optimizer.state_dict(),
                                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict()))
            restore_adapter(model, before)
            state = load_checkpoint(path)
            restore_adapter(model, state["adapter"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
        report["models"][name] = dict(**info, embedding_shape=list(embeddings.shape),
                                     loss=loss, dtype=str(dtype), backward=True,
                                     accumulation_steps=2, checkpoint_roundtrip=True)
        print(f"PASS {name}: {info['trainable_parameters']:,} trainable; loss={loss:.5f}", flush=True)
        del model, optimizer, scheduler, loader, tokens, embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["passed"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(f"Saved verification: {args.report}")


if __name__ == "__main__":
    main()
