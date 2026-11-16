# Evaluate a training run

`evaluation.ipynb` displays a completed training run's configuration, saved metrics, loss and validation AUC history, and evaluates its saved adapter on the validation and test pairs. It plots ROC curves, score distributions, and confusion matrices using the threshold selected on validation during training.

From the repository root, install the training dependencies and Matplotlib in the notebook environment:

```bash
python -m pip install -r 3_train/requirements.txt matplotlib
python -m jupyterlab 4_evaluation/evaluation.ipynb
```

Set `RUN_DIR` in the notebook to a completed output directory containing `config.json`, `metrics.json`, `history.csv`, `validation_pairs.csv`, `test_pairs.csv`, and `adapter/`. If the dataset has moved, set `DATA_PATH` to the original Parquet dataset. Run all cells in order. The base model checkpoint must be available locally or downloadable from Hugging Face. Inference can take time, particularly on CPU.

Scores use the training notebook's text formatting, tokenization, masked mean pooling, normalization and batch size. Mixed precision is used on CUDA if the original run used it; differences in hardware or precision may affect results, especially at the classification threshold. The notebook does not change the run directory or save predictions.
