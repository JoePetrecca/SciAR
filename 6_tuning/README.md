# Quick SciBERT tuning

This is a deliberately small Optuna search for the SciBERT author-similarity
encoder. The notebook is based directly on `3_train/train_lora.ipynb`: it
uses the same author split, SPECTER pair construction, encoder, pooling,
cosine loss, and validation evaluation. Its only material change is that
Optuna repeats the training cell with sampled hyperparameters.

It creates topic-matched **train** and **validation** pairs, then tunes for
validation ROC-AUC. It never creates, loads, or scores test pairs.

Install the training dependencies plus Optuna:

```bash
python -m pip install -r 6_tuning/requirements.txt
```

Run 25 pruned trials:

```bash
python -m jupyterlab 6_tuning/tune_scibert.ipynb
```

Set `N_TRIALS` in the configuration cell, then run all cells. Outputs are
written to `6_tuning/outputs/scibert_quick/`, including pair manifests, a
resumable `study.db`, and `best_trial.json`. The notebook uses 20,000 training
pairs, 2,000 validation pairs, and three epochs per trial; these settings are
intentionally cheaper than the full training run.

After tuning, copy the winning parameters into a fresh full `3_train` SciBERT
run. Select its epoch and threshold on validation, then evaluate test once.

## Four-GPU workers

`tune_worker.py` runs one trial at a time on its visible GPU. First create the
fixed train/validation pair CSVs once, then start one worker per GPU:

```bash
python 6_tuning/tune_worker.py --prepare
CUDA_VISIBLE_DEVICES=0 python 6_tuning/tune_worker.py --trials 7 &
CUDA_VISIBLE_DEVICES=1 python 6_tuning/tune_worker.py --trials 7 &
CUDA_VISIBLE_DEVICES=2 python 6_tuning/tune_worker.py --trials 7 &
CUDA_VISIBLE_DEVICES=3 python 6_tuning/tune_worker.py --trials 7 &
wait
```

The workers share `study.db`; SQLite is suitable for this small four-worker
run, but do not run more workers than this or start a second batch at once.
