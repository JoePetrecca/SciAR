"""Aggregate every seed; paired component bootstrap separates sampling from seed variation."""
import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import HERE, configuration, experiments, read_json, write_json


def author_components(frame):
    """Connect all pairs sharing any author (thus also any consistently labeled paper)."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(frame.author_a, frame.author_b):
        a, b = find(a), find(b)
        parent[b] = a
    roots = [find(a) for a in frame.author_a]
    return pd.factorize(np.asarray(roots), sort=True)[0]


def paired_bootstrap(frame, left, right, repetitions=1000, seed=67):
    """Resample connected author components, using identical weights for both models/seeds."""
    components = author_components(frame)
    counts = np.bincount(components)
    n = len(counts)
    base = float(np.mean([roc_auc_score(frame.label, a) - roc_auc_score(frame.label, b)
                          for a, b in zip(left, right)]))
    result = dict(difference=base, components=n, largest_component_fraction=float(counts.max()/len(frame)),
                  repetitions=repetitions, valid_replicates=0, interval_95=None,
                  estimand="mean of paired per-seed AUC differences; training seeds held fixed")
    if n < 2:
        result["warning"] = "One connected component: cluster sampling uncertainty is not estimable"
        return result
    if n < 30 or counts.max()/len(frame) > .2:
        result["warning"] = "Few or dominant components: confidence interval may be unreliable"
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        weights = rng.multinomial(n, np.full(n, 1/n))[components]
        if any(weights[frame.label.to_numpy() == y].sum() == 0 for y in (0, 1)):
            continue
        differences.append(np.mean([
            roc_auc_score(frame.label, a, sample_weight=weights) -
            roc_auc_score(frame.label, b, sample_weight=weights)
            for a, b in zip(left, right)]))
    result["valid_replicates"] = len(differences)
    if differences:
        result["interval_95"] = np.quantile(differences, [.025, .975]).tolist()
    return result


def collect(root, destination, repetitions=1000, allow_incomplete=False):
    models, config = configuration()
    rows, histories, scores = [], [], {}
    fingerprint = None
    reference = None
    for model, seed in experiments():
        run = root / model / f"seed_{seed}"
        if not (run / "metrics.json").exists():
            if allow_incomplete:
                warnings.warn(f"Missing run: {model}/{seed}")
                continue
            raise ValueError(f"Missing completed run: {run}; use --allow-incomplete for diagnostics only")
        metadata = read_json(run / "metrics.json")
        run_config = read_json(run / "config.json")
        if run_config.get("cpu_smoke", False) and not allow_incomplete:
            raise ValueError("CPU smoke outputs cannot be reported as the full study")
        if metadata["model"] != model or metadata["seed"] != seed:
            raise ValueError("Output directory/run identity mismatch")
        if fingerprint is None:
            fingerprint = metadata["dataset_fingerprint"]
        if fingerprint != metadata["dataset_fingerprint"]:
            raise ValueError("Cannot compare different datasets")
        test = pd.read_csv(run / "finetuned_test_scores.csv",
                           dtype={"pair_id": str, "author_a": str, "author_b": str,
                                  "bibcode_a": str, "bibcode_b": str})
        columns = ["pair_id", "bibcode_a", "bibcode_b", "author_a", "author_b", "label"]
        if test.pair_id.duplicated().any():
            raise ValueError("Duplicate pair IDs")
        test = test.sort_values("pair_id").reset_index(drop=True)
        if reference is None:
            reference = test[columns]
        if not reference.equals(test[columns]):
            raise ValueError("Runs do not contain exactly the same test pairs and labels")
        scores[(model, seed)] = test.score.to_numpy()
        m = metadata["finetuned"]["test"]
        rows.append(dict(model=model, seed=seed, test_auc=m["roc_auc"], accuracy=m["accuracy"],
                         precision=m["precision"], recall=m["recall"], f1=m["f1"],
                         baseline_auc=metadata["baseline"]["test"]["roc_auc"],
                         validation_auc=metadata["best_validation_auc"],
                         best_epoch=metadata["best_epoch"], training_seconds=metadata["training_seconds"]))
        history = pd.read_csv(run / "history.csv")
        history["model"], history["seed"] = model, seed
        histories.append(history)
    if not rows:
        raise ValueError("No completed runs")
    destination.mkdir(parents=True, exist_ok=True)
    individual = pd.DataFrame(rows)
    individual.to_csv(destination / "individual_runs.csv", index=False)
    summary = individual.groupby("model", sort=False).agg(
        runs=("seed", "count"), mean_test_auc=("test_auc", "mean"),
        std_test_auc=("test_auc", "std"), mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"), mean_f1=("f1", "mean"), std_f1=("f1", "std"),
        mean_baseline_auc=("baseline_auc", "mean"))
    summary.to_csv(destination / "summary.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.index, summary.mean_test_auc, yerr=summary.std_test_auc.fillna(0), capsize=4)
    ax.set(ylabel="Test ROC-AUC (mean ± sample SD across seeds)", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(destination / "mean_auc.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for history in histories:
        ax.plot(history.epoch, history.val_auc, alpha=.7,
                label=f"{history.model.iloc[0]} / {history.seed.iloc[0]}")
    ax.set(xlabel="Epoch", ylabel="Validation ROC-AUC")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(destination / "validation_comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    ax.bar(x-.2, summary.mean_baseline_auc, .4, label="Pretrained")
    ax.bar(x+.2, summary.mean_test_auc, .4, label="Fine-tuned")
    ax.set(xticks=x, xticklabels=summary.index, ylabel="Mean test ROC-AUC", ylim=(0, 1))
    ax.legend()
    fig.tight_layout(); fig.savefig(destination / "baseline_comparison.png", dpi=160); plt.close(fig)
    comparisons = {}
    for a, b in itertools.combinations(models, 2):
        # Never select or rank individual seeds by their test performance.
        seeds = config["seeds"]
        if all((m, s) in scores for m in (a, b) for s in seeds):
            comparisons[f"{a}_minus_{b}"] = paired_bootstrap(
                reference, [scores[(a, s)] for s in seeds], [scores[(b, s)] for s in seeds],
                repetitions)
    write_json(destination / "bootstrap.json", dict(
        dataset_fingerprint=fingerprint, comparisons=comparisons,
        method="Paired connected-author-component percentile bootstrap",
        caveats="Seeds fixed, not a training-population CI. No multiplicity correction or significance claim. "
                "Depends on independent components; topic and undetected duplicates may violate this."))
    print(summary.to_string())
    return individual, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=HERE / "outputs")
    parser.add_argument("--destination", type=Path, default=HERE / "outputs/comparison")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be >=1")
    collect(args.outputs, args.destination, args.bootstrap, args.allow_incomplete)
