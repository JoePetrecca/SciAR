"""Export scores/metrics/plots; CLI replots saved results without model inference."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from common import choose_threshold, metrics, read_json, write_json


def export_scores(directory, prefix, pairs, evaluations):
    threshold = choose_threshold(evaluations["validation"][1], evaluations["validation"][2])
    result = {}
    for split in ("validation", "test"):
        loss, labels, scores = evaluations[split]
        frame = pairs[split].copy()
        if not np.array_equal(frame.label, labels):
            raise ValueError("Evaluation order does not match cached pairs")
        frame["score"] = scores
        frame["prediction"] = (scores >= threshold).astype(int)
        frame.to_csv(Path(directory) / f"{prefix}_{split}_scores.csv", index=False)
        result[split] = dict(**metrics(labels, scores, threshold), loss=loss)
    write_json(Path(directory) / f"{prefix}_metrics.json", result)
    return result


def plots(directory):
    directory = Path(directory)
    history = pd.read_csv(directory / "history.csv")
    def finish(name):
        plt.tight_layout()
        plt.savefig(directory / f"{name}.png", dpi=160)
        plt.close()
    plt.figure()
    for key in ("train_loss", "val_loss"):
        plt.plot(history.epoch, history[key], label=key)
    plt.xlabel("Epoch"); plt.ylabel("Cosine embedding loss"); plt.legend()
    finish("loss")
    plt.figure()
    plt.plot(history.epoch, history.val_auc)
    plt.xlabel("Epoch"); plt.ylabel("Validation ROC-AUC")
    finish("validation_auc")
    test = pd.read_csv(directory / "finetuned_test_scores.csv")
    plt.figure()
    for prefix in ("baseline", "finetuned"):
        frame = pd.read_csv(directory / f"{prefix}_test_scores.csv")
        fpr, tpr, _ = roc_curve(frame.label, frame.score)
        plt.plot(fpr, tpr, label=prefix)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
    finish("test_roc")
    plt.figure()
    for label, name in ((0, "Different author"), (1, "Same author")):
        plt.hist(test.loc[test.label == label, "score"], bins=40, alpha=.5, label=name)
    plt.xlabel("Cosine similarity"); plt.ylabel("Pairs"); plt.legend()
    finish("test_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, split in zip(axes, ("validation", "test")):
        frame = pd.read_csv(directory / f"finetuned_{split}_scores.csv")
        matrix = confusion_matrix(frame.label, frame.prediction, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), n in np.ndenumerate(matrix):
            ax.text(j, i, str(n), ha="center", va="center")
        ax.set(title=split, xlabel="Predicted (0=different, 1=same)",
               ylabel="Actual", xticks=[0, 1], yticks=[0, 1])
    finish("confusion_matrices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    plots(args.run)
    print(read_json(args.run / "finetuned_metrics.json"))
