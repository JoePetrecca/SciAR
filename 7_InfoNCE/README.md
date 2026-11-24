# InfoNCE experiments

`train_infonce.py` keeps the existing author-disjoint split and matched-pair
validation/test evaluation, while training paper embeddings with symmetric
in-batch InfoNCE.

Each training batch contains 16 distinct authors and two papers per author.
The two papers from an author form the positive pair; papers from the other
15 authors are in-batch negatives. The defaults use temperature `0.07`, LoRA,
and early stopping with validation AUC patience of two epochs.

Run one experiment from the project root:

```bash
python3 7_InfoNCE/train_infonce.py \
  "allenai/scibert_scivocab_uncased" \
  "SciAR_scibert_0" \
  0
```

Use `python3 7_InfoNCE/run_experiments.py` to generate and submit five Slurm
runs for every configured model.

Results are written to `7_InfoNCE/outputs/<run-name>/`.
