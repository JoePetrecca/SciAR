"""Build pairs once, using the existing notebook reader and SPECTER sampler."""
import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np

from common import (HERE, ROOT, SPLITS, configuration, load_dataset, read_json,
                    save_manifest, sha256, validate_pairs, write_json)


def existing_readers():
    namespace = {}
    for cell in read_json(ROOT / "2_Data/data_reader.ipynb")["cells"]:
        if cell["cell_type"] == "code" and "reader-code" in cell.get("metadata", {}).get("tags", []):
            exec("".join(cell["source"]), namespace)
    spec = importlib.util.spec_from_file_location("specter_reader", ROOT / "2_Data/specter_reader.py")
    specter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(specter)
    return namespace, specter


def subset_pairs(frame, size):
    """Keep positive/negative topic matches together; independent of requested size."""
    if size < 2 or size % 2 or size > len(frame):
        raise ValueError("Subset size must be positive, even, and <= parent size")
    ids = sorted(frame.match_id.unique(), key=lambda x: __import__("hashlib").sha256(
        f"67:{x}".encode()).hexdigest())
    result = frame[frame.match_id.isin(ids[:size//2])].copy().reset_index(drop=True)
    if len(result) != size:
        raise ValueError("Parent does not have two rows per match")
    return result


def enrich_pairs(frame, papers, split):
    frame = frame.drop(columns=["text_a", "text_b"])
    lookup = papers.set_index("bibcode").first_author_orcid
    for side in ("a", "b"):
        frame[f"author_{side}"] = frame[f"bibcode_{side}"].map(lookup)
    frame["pair_id"] = [f"{split}:{i}" for i in range(len(frame))]
    return frame


def main():
    _, config = configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=ROOT / "data/processed/sciar_refereed_v1/papers.parquet")
    parser.add_argument("--output", type=Path, default=HERE / "data/fixed_100k")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/specter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16",
                        help="SPECTER encoding precision; float16 reuses the existing GPU cache")
    parser.add_argument("--train-pairs", type=int, default=100000)
    parser.add_argument("--validation-pairs", type=int, default=2000)
    parser.add_argument("--test-pairs", type=int, default=5000)
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument("--subset-of", type=Path)
    parent_group.add_argument("--extend-from", type=Path,
                              help="Freeze parent author assignments and validation/test, resample larger train")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    sizes = dict(train=args.train_pairs, validation=args.validation_pairs, test=args.test_pairs)
    if any(n < 2 or n % 2 for n in sizes.values()):
        parser.error("All split sizes must be positive and even")
    if args.output.exists():
        parser.error("Output already exists; use a new directory (never overwrite frozen data)")
    parent_path = args.subset_of or args.extend_from
    if parent_path:
        pairs, papers, parent = load_dataset(parent_path)
        if sizes["validation"] != parent["sizes"]["validation"] or sizes["test"] != parent["sizes"]["test"]:
            parser.error("Subsets must retain the parent's validation/test sizes")
        splits = read_json(parent_path / "author_splits.json")
        if args.subset_of:
            pairs["train"] = subset_pairs(pairs["train"], sizes["train"])
        else:
            import torch
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for SPECTER preparation, but unavailable")
            if sizes["train"] <= parent["sizes"]["train"]:
                parser.error("--extend-from requires a larger training size; use --subset-of to shrink")
            _, specter = existing_readers()
            vectors = specter.specter_embeddings(papers, args.cache, device=args.device,
                                               precision=args.precision, batch_size=args.batch_size)
            frame = specter.make_matched_pairs(
                papers, vectors, splits["train"], sizes["train"], seed=67,
                paper_limit=2, author_limit=10, device=args.device, batch_size=16)
            pairs["train"] = enrich_pairs(frame, papers, "train")
        metadata = dict(parent_fingerprint=parent["fingerprint"], dataset_seed=67, sizes=sizes)
        metadata["operation"] = "subset" if args.subset_of else "extend_train_fixed_evaluation"
        if args.extend_from:
            metadata["precision"] = args.precision
        metadata["sampler_sha256"] = sha256(ROOT / "2_Data/specter_reader.py")
    else:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for SPECTER preparation, but unavailable")
        reader, specter = existing_readers()
        all_papers = reader["load_papers"](args.papers)
        if not all_papers.bibcode.is_unique:
            raise ValueError("Input bibcodes are not unique")
        keep, dedup = specter.deduplicate_works(all_papers)
        papers = all_papers.iloc[keep].reset_index(drop=True)
        splits = reader["split_authors"](papers, seed=67)
        vectors = specter.specter_embeddings(all_papers, args.cache, device=args.device,
                                           precision=args.precision, batch_size=args.batch_size)[keep]
        pairs = {}
        for offset, split in enumerate(SPLITS):
            frame = specter.make_matched_pairs(
                papers, vectors, splits[split], sizes[split], seed=67+offset,
                paper_limit=2, author_limit=10, device=args.device, batch_size=16)
            pairs[split] = enrich_pairs(frame, papers, split)
        metadata = dict(dataset_seed=config["dataset_seed"], split_seeds=[67, 68, 69],
                        sizes=sizes, source=str(args.papers.resolve()),
                        source_sha256=sha256(args.papers), dedup=dedup,
                        method="matched_distinct_v1", precision=args.precision,
                        paper_limit=2, author_limit=10,
                        reader_sha256=sha256(ROOT / "2_Data/data_reader.ipynb"),
                        sampler_sha256=sha256(ROOT / "2_Data/specter_reader.py"))
    validate_pairs(pairs, papers, splits)
    args.output.mkdir(parents=True, exist_ok=False)
    papers.to_parquet(args.output / "papers.parquet", index=False)
    write_json(args.output / "author_splits.json", splits)
    for split, frame in pairs.items():
        frame.to_parquet(args.output / f"{split}.parquet", index=False)
    if parent_path:
        # Keep evaluation files, text corpus and author assignments byte-identical.
        for filename in ("papers.parquet", "author_splits.json", "validation.parquet", "test.parquet"):
            shutil.copyfile(parent_path / filename, args.output / filename)
    save_manifest(args.output, metadata)
    _, _, manifest = load_dataset(args.output)
    print(f"Frozen dataset: {args.output}\nFingerprint: {manifest['fingerprint']}")


if __name__ == "__main__":
    main()
