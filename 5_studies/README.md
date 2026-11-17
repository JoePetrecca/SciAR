# SciAR: fixed-pair, five-encoder study

Compare pretrained encoders for first-author verification using titles and abstracts.
This directory is independent of `3_train/` and does not modify its experiments.
No training jobs are submitted by the Python programs.

## Design

| Key | Pretrained checkpoint | LoRA targets |
|---|---|---|
| bert | `bert-base-uncased` | attention query/value |
| scibert | `allenai/scibert_scivocab_uncased` | attention query/value |
| roberta | `roberta-base` | attention query/value |
| specter | `allenai/specter` | attention query/value |
| modernbert | `answerdotai/ModernBERT-base` | attention fused `Wqkv` |

Each uses seeds **67, 137, 2026**: 15 independent runs, model-major order in
`config/models.json`, seed-minor order in `config/training.json`. Tasks 0–2 are
BERT, 3–5 SciBERT, 6–8 RoBERTa, 9–11 SPECTER, 12–14 ModernBERT.
LUAR is deliberately excluded. ModernBERT's fused projection adapts Q, K **and**
V, so its adapter parameter count differs; this is an architectural difference,
not an identical-adapter-budget experiment.

Defaults: 100,000/2,000/5,000 balanced train/validation/test pairs; effective batch
16 pairs (32 papers); at most 10 epochs; patience 3; AdamW LR 0.0002,
weight decay 0.01; rank 8, alpha 16, adapter dropout 0.1; margin 0.2;
maximum 512 tokens/document. A constant LR scheduler preserves the existing
training approach and has saved/restored state.

The encoder sees only `title + "\n\n" + abstract`. All nonpadding token states,
including special tokens, are mean-pooled in float32 and L2-normalized.
CosineEmbeddingLoss receives +1/-1; stored labels remain 1/0.
Highest **validation ROC-AUC** selects the checkpoint; strict ties do not improve.
After three nonimproving epochs, training stops. The best adapter is restored.
Validation Youden J (TPR−FPR) chooses the threshold, equivalent to maximizing
balanced accuracy. Each baseline has its own validation-only threshold.
Test inference occurs for the pretrained baseline and after final restoration,
**never during epoch selection**. All seeds are reported; do not select a seed
or tune hyperparameters using these test outputs.

## Environment

Use Python 3.11+ with a **CUDA-enabled** PyTorch build compatible with the allocated
node's driver, then install:

```bash
python -m pip install -r 5_studies/requirements.txt
mkdir -p 5_studies/logs
```

Do not install into someone else's shared environment. Activate your environment
before `sbatch`, or export an absolute `SCIAR_PYTHON` path. For the environment
found during implementation:

```bash
export SCIAR_PYTHON=/mnt/home/petrecc1/miniforge3/envs/SciARvenv/bin/python
```

Requirements specify compatibility ranges; each run records actual PyTorch,
Transformers, PEFT, CUDA and GPU versions. Save `python -m pip freeze` alongside
the study for archival reproducibility. Pretrained models/tokenizers require
Hugging Face access once. Resolved model commit hashes are recorded in
`parameters.json`; pin `"revision": "<commit>"` in each model spec for a frozen
long-term study. Resume rejects changed resolved revisions. Eager attention is
used for all architectures, avoiding optional FlashAttention dependencies.

## 1. Prepare the dataset once

Run preparation as a separate operation in an approved GPU allocation. This
encodes the corpus using the **existing** SPECTER implementation; it may be
expensive and is not part of the 15 training tasks.

```bash
python 5_studies/prepare_data.py \
  --papers /mnt/research/deepthought/SciAR/data/processed/sciar_refereed_v1/papers.parquet \
  --output 5_studies/data/fixed_100k --device cuda
```

Or set the correct source path on another filesystem. The default source is
`data/processed/sciar_refereed_v1/papers.parquet` relative to the repository.
`--cache` defaults to the existing `data/cache/specter/`. Preparation uses one
device and FP16 SPECTER CLS encoding by default, reusing the existing compatible
FP16 cache. Use `--precision float32` for full-precision encoding or CPU preparation.
The chosen encoding precision is recorded in the dataset manifest. Cached vectors
and similarity matching use float32 storage/arithmetic even when encoding uses FP16.
Do not run concurrent preparations against the same SPECTER cache.

`prepare_data.py` executes only the tagged reader definitions from
`2_Data/data_reader.ipynb`, then directly calls `deduplicate_works`,
`specter_embeddings`, and `make_matched_pairs` in `2_Data/specter_reader.py`.
It preserves the existing 80/10/10 author split, exact-title/abstract deduplication,
conflicting-author removal, candidate near-duplicate guards, distinct positives,
and nearest-topic-similarity negative matching. Shared limits remain two paper
appearances and ten author pair appearances. It fails rather than silently
relaxing limits if quotas cannot be filled.

Dataset-generation seed is **67**, with deterministic split pair seeds 67/68/69
as in the existing notebook. Training seeds never enter preparation.
The cache contains:

* `papers.parquet`: paper ID, first-author ORCID, original title/abstract.
* `{train,validation,test}.parquet`: pair IDs, both paper/author IDs, label,
  matched-topic ID and SPECTER similarity.
* `author_splits.json`: complete author assignments, including unused authors.
* `manifest.json`: sizes, generation settings, dedup stats, source/sampler hashes,
  file hashes and a SHA-256 dataset fingerprint.

Preparation refuses an existing output directory; the manifest is written last.
If preparation fails before completion, inspect the partial directory and choose
a new output path. Existing SPECTER shards remain reusable.
Training rechecks every cached file hash, balance, labels, uniqueness and
author/paper disjointness before loading a model. It **never samples pairs**.
These filters preserve the existing implementation, not exhaustive corpus-wide
near-duplicate detection: undetected cross-split near-duplicates may remain.
ORCIDs represent first authors, not verified text writers.

## 2. GPU preflight — required before the array

Obtain approval before requesting resources. Example MSU interactive allocation:

```bash
salloc --partition=general-short --gpus=a100:1 --ntasks=1 \
  --cpus-per-task=4 --mem=48G --time=01:00:00
```

Inside that allocation, from the repository root and with the environment active:

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python 5_studies/verify.py --report 5_studies/verification/gpu.json
```

This downloads/initializes **all five actual pretrained models and tokenizers**,
asserts expected per-layer LoRA attachment, checks embedding shape/unit norms,
runs forward/backward, checks nonzero LoRA gradients and parameter/input device,
tests AMP and two-step accumulation, confirms adapter updates and round-trips
adapter/optimizer/scheduler/scaler checkpoints. BF16 is automatic when supported,
otherwise FP16 with a gradient scaler. No CPU fallback is allowed.

The report must be a successful CUDA run inside SLURM. The array checks its
configuration and implementation hashes; CPU reports cannot authorize GPU jobs.
Rerun after code/config changes. The preflight uses short sequences and tiny
microbatches: it verifies mechanics, **not** full-length memory capacity or
throughput. Test your intended microbatch at length 512 in an approved pilot
before committing resources to the array. A successful report is a technical
gate, **not user approval to submit jobs**.

## 3. Run or submit (only after approval)

One full run inside an allocated GPU:

```bash
python 5_studies/train.py --model bert --seed 67 --data 5_studies/data/fixed_100k
```

All 15 runs, initially at most two concurrent GPUs:

```bash
mkdir -p 5_studies/logs
sbatch --array=0-14%2 5_studies/submit.sbatch
```

The script requests one A100, four CPU cores, 48 GB host RAM, and 24 hours per
task in `general-long-gpu`. Resource syntax was checked against
[MSU GPU requests](https://docs.icer.msu.edu/Requesting_GPUs/) and live `sinfo`.
No guessed account is specified; pass `--account=...` only if your allocation
requires it. These requests are starting points, not measured runtime guarantees.
GPU type may be overridden at submission (`--gpus=h200:1`, for example).
Do not override array indices without updating the model/seed mapping.
SLURM log directories must exist **before** submission.

Optional environment variables: `SCIAR_DATA`, `SCIAR_PYTHON`, `SCIAR_PREFLIGHT`,
`SCIAR_MICROBATCH`, `SCIAR_GRADIENT_CHECKPOINTING=1`, `SCIAR_RESUME=1`.
No script changes `CUDA_VISIBLE_DEVICES`. PyTorch uses its allocated device 0.
GPU utilization/memory are sampled by `nvidia-smi` every 30 seconds under `logs/`.
PyTorch peak allocated memory, GPU capacity/name, SLURM IDs, CUDA/software
versions, best epoch, epoch timings and active runtime are saved in run metrics.
Utilization monitor CSVs are separate from PyTorch allocator measurements.

### Memory and recovery

```bash
python 5_studies/train.py --model modernbert --seed 67 \
  --microbatch-size 4 --gradient-checkpointing
```

This preserves effective batch 16 via four accumulation steps. Microbatch must
divide effective batch exactly. Final partial windows are weighted by actual
pair count. `--workers` controls pinned-memory DataLoaders; default two workers.
Tokenization occurs on CPU; each combined pair batch is transferred once, then
encoded in one shared-encoder forward pass. AMP, microbatch and accumulation
settings are recorded per run; report deviations in study comparisons.

```bash
python 5_studies/train.py --model bert --seed 67 --resume
# Resume array tasks, using the same settings as the initial submission:
SCIAR_RESUME=1 sbatch --array=0-14%2 5_studies/submit.sbatch
```

`last.pt` is an atomic **epoch-boundary** checkpoint containing current/best
adapters, optimizer, scheduler, AMP scaler, RNG states (Python, NumPy, PyTorch,
CUDA and training-loader generator), history and patience counters. An interrupted
partial epoch is replayed, not resumed midbatch. A run interrupted before its
first checkpoint restarts its first epoch. `best.pt` is separately saved.
Completed runs are skipped only with `--resume`; otherwise existing output
directories are protected. Trusted local PyTorch checkpoints only: loading
untrusted pickle checkpoints is unsafe.

Resume rejects changed data, code, model revision, seed or training configuration.
To change microbatch after an OOM, start a new output root and record the change;
do not rewrite provenance to force a resume. CUDA OOM reports memory remedies
without silently changing the experiment. Reproducibility is strongest on the
same hardware/software; GPU kernels are not guaranteed bitwise deterministic
across environments.

## Outputs and evaluation

Each `outputs/<model>/seed_<seed>/` contains:

* `config.json`, `parameters.json`, `history.csv`, `metrics.json`.
* `last.pt`, `best.pt`, `adapter/` (PEFT weights plus tokenizer).
* `baseline_metrics.json`, `finetuned_metrics.json`.
* `{baseline,finetuned}_{validation,test}_scores.csv`: pair and author IDs,
  labels, cosine scores and predictions.
* `loss.png`, `validation_auc.png`, `test_roc.png`,
  `test_distributions.png`, `confusion_matrices.png`.

Classification exports include ROC-AUC, accuracy, precision, recall, F1, threshold
and a 2×2 confusion matrix (row=actual, column=predicted, order different/same).
Metrics also contain losses, best epoch and training duration. Completion
`metrics.json` is written only after successful export and plotting.
Base weights remain in the Hugging Face cache; adapters are not standalone
merged models. The model revision must remain available.

Replot without model inference:

```bash
python 5_studies/evaluate.py 5_studies/outputs/bert/seed_67
python 5_studies/collect_results.py --bootstrap 1000
```

Aggregation requires all 15 runs, identical dataset fingerprints and exactly
aligned test pairs. `--allow-incomplete` is for debugging, not final reporting.
`outputs/comparison/` includes `individual_runs.csv`, `summary.csv` (mean and
**sample SD, ddof=1**, across seeds), mean-AUC error bars, validation trajectories,
baseline comparison and `bootstrap.json`. Open `analyze_results.ipynb` to inspect.
There is no test-based seed ranking.

### Paired bootstrap

An ordinary independent-pair bootstrap is invalid when papers/authors recur.
Instead we form connected components of the test **author graph**: each pair
connects its two authors; positive pairs form self edges. Pairs sharing an author
or a paper stay in the same component, transitively. For each replicate, sample
the same number of components with replacement, assigning their multiplicities
as pair weights. Use identical weights for both compared models and all seeds.
Compute each seed's weighted AUC difference and average those differences over
the three **fixed** matched seeds. Report the 2.5/97.5 percentile interval.
This is not AUC of averaged scores (an ensemble).

This estimates **test-sampling uncertainty conditional on the three trained
seeds**; across-seed sample SD is reported separately. Components are assumed
independent. Topic dependencies and undetected duplicate works can violate this.
Negative edges may produce few/giant components; counts and largest-component
fraction are reported, warnings flag <30 components or >20% dominance, and no
interval is returned for a single component. Replicates lacking either class
are excluded and counted. Fixed-pair sampling and topic matching further limit
generalization. No multiplicity correction, p-values, or automatic significance
claims are made; three seeds alone do not establish significance.

## Future scaling and extension

`--train-pairs`, `--validation-pairs`, `--test-pairs` make sizes configurable.
For nested 20k/50k/100k/250k/500k studies, **first** prepare a sufficiently large
parent with the desired author assignments, then derive all smaller versions:

```bash
# Future work only — do not launch as part of the initial study:
python 5_studies/prepare_data.py --subset-of PATH_TO_FROZEN_PARENT \
  --train-pairs 20000 --output 5_studies/data/fixed_20k
```

Subsets retain byte-identical validation/test pairs and author assignments.
Training keeps deterministic hash-ordered positive/negative match groups, so
smaller versions derived from the **same parent** are nested and balanced.
Separate fresh preparations at different sizes are **not** guaranteed nested.
The existing diversity limits may make 500k infeasible for this corpus; the
sampler will fail explicitly. A 100k cache cannot supply a larger subset.
To construct a larger parent **without changing this study's author assignments,
paper corpus or validation/test files**, use:

```bash
# Future work only:
python 5_studies/prepare_data.py --extend-from 5_studies/data/fixed_100k \
  --train-pairs 500000 --output 5_studies/data/parent_500k --device cuda
```

This resamples training only, preserving the existing matching/diversity limits;
the old 100k train is not guaranteed to be a subset of the new parent. Derive
all nested scaling sizes from that one new parent with `--subset-of`. Both
operations copy the original evaluation and author-assignment files verbatim.

Add a model to `models.json` with checkpoint, target suffixes and expected target
count per layer; extend `engine.load_encoder` for any architecture needing special
handling. Failed attachment is an error. Add seeds to `training.json`; update the
SLURM array range and rerun preflight. LUAR will need a dedicated encoder/pooling
integration rather than assuming it is an AutoModel-compatible drop-in.

## Tests and verification status

```bash
python -m unittest discover -s 5_studies/tests -v
python 5_studies/verify.py --cpu --report 5_studies/verification/cpu.json
```

CPU smoke training requires `--cpu-smoke`, <=256 pairs per split and <=2 epochs.
Use a small frozen cache and a JSON override with `epochs: 2`, `workers: 0`.
It is not a substitute for SLURM/CUDA verification.

During implementation the actual five pretrained encoders/tokenizers passed CPU
forward/backward, LoRA attachment, unit-norm embeddings, loss, two-step
accumulation, optimizer update and checkpoint round-trips. Observed adapter
counts: 294,912 for each BERT-family/RoBERTa model; 540,672 for ModernBERT.
The offline tests also cover cached SPECTER preparation, nested subsets,
all 15 task mappings/shared dataset hashes, tamper/leakage checks,
best-checkpoint/early-stopping logic, interruption/resumption against an
uninterrupted tiny-model run, score/plot exports, aggregation with 15 synthetic
run outputs and component bootstrap edge cases.

**Not yet verified:** CUDA AMP/backward inside a SLURM allocation, full-length
memory/throughput, scheduler execution of the array, and full-corpus 107,000-pair
preparation. No full dataset, training array or scaling jobs were launched.
The GPU preflight remains required; the CPU report cannot satisfy that gate.
