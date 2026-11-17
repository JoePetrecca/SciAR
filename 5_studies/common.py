"""Shared, model-independent dataset integrity and study configuration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = ("train", "validation", "test")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration():
    return read_json(HERE / "config/models.json"), read_json(HERE / "config/training.json")


def experiments():
    models, config = configuration()
    return [(model, seed) for model in models for seed in config["seeds"]]


def select_experiment(task):
    runs = experiments()
    if not 0 <= task < len(runs):
        raise ValueError(f"Array index must be between 0 and {len(runs)-1}")
    return runs[task]


def validate_pairs(pairs, papers, author_splits):
    if not papers.bibcode.is_unique or papers.isna().any().any():
        raise ValueError("Paper identifiers must be unique and metadata non-null")
    lookup = papers.set_index("bibcode").first_author_orcid
    used_papers, used_authors = set(), set()
    assigned = set()
    for split in SPLITS:
        authors = set(author_splits[split])
        if assigned & authors:
            raise ValueError("Overlapping author assignments")
        assigned |= authors
        frame = pairs[split]
        if frame.empty or frame.isna().any().any() or not frame.pair_id.is_unique:
            raise ValueError(f"Invalid/empty {split} pairs")
        if frame.label.value_counts().to_dict() != {0: len(frame)//2, 1: len(frame)//2}:
            raise ValueError(f"Unbalanced {split} pairs")
        keys = [tuple(sorted(x)) for x in zip(frame.bibcode_a, frame.bibcode_b)]
        if len(set(keys)) != len(keys) or (frame.bibcode_a == frame.bibcode_b).any():
            raise ValueError("Repeated or self pairs")
        current_papers, current_authors = set(), set()
        for side in ("a", "b"):
            ids = frame[f"bibcode_{side}"]
            expected = ids.map(lookup)
            if expected.isna().any() or not expected.equals(frame[f"author_{side}"]):
                raise ValueError("Missing papers or inconsistent author metadata")
            current_papers.update(ids)
            current_authors.update(expected)
        if not current_authors <= authors:
            raise ValueError(f"Authors outside {split} assignment")
        if not np.array_equal((frame.author_a == frame.author_b).astype(int), frame.label):
            raise ValueError("Incorrect authorship labels")
        if used_papers & current_papers or used_authors & current_authors:
            raise ValueError("Paper or author leakage across splits")
        used_papers |= current_papers
        used_authors |= current_authors


def save_manifest(directory, metadata):
    directory = Path(directory)
    files = ["papers.parquet", "author_splits.json"] + [f"{s}.parquet" for s in SPLITS]
    metadata["files"] = {f: sha256(directory / f) for f in files}
    metadata["fingerprint"] = canonical_hash(metadata)
    write_json(directory / "manifest.json", metadata)


def load_dataset(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if canonical_hash(payload) != manifest["fingerprint"]:
        raise ValueError("Dataset manifest fingerprint mismatch")
    required = {"papers.parquet", "author_splits.json", *(f"{s}.parquet" for s in SPLITS)}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete dataset manifest")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Dataset content changed: {name}")
    papers = pd.read_parquet(directory / "papers.parquet")
    pairs = {s: pd.read_parquet(directory / f"{s}.parquet") for s in SPLITS}
    validate_pairs(pairs, papers, read_json(directory / "author_splits.json"))
    if {s: len(f) for s, f in pairs.items()} != manifest["sizes"]:
        raise ValueError("Dataset sizes differ from manifest")
    return pairs, papers, manifest


def choose_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr-fpr)[valid])])


def metrics(labels, scores, threshold):
    pred = np.asarray(scores) >= threshold
    return dict(roc_auc=float(roc_auc_score(labels, scores)),
                accuracy=float(accuracy_score(labels, pred)),
                precision=float(precision_score(labels, pred, zero_division=0)),
                recall=float(recall_score(labels, pred, zero_division=0)),
                f1=float(f1_score(labels, pred, zero_division=0)),
                confusion_matrix=confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
                threshold=float(threshold))
