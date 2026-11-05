# BERT LoRA training

`train_lora.ipynb` learns author-comparison embeddings from paper titles and abstracts using a shared BERT encoder and cosine contrastive loss (margin `0.2`). LoRA updates only attention query/value adapters.

## Run

From the repository root, install dependencies in your notebook environment:

```bash
python -m pip install -r 3_train/requirements.txt
python -m jupyterlab 3_train/train_lora.ipynb
```

Select that environment's kernel. Review **Configuration**, then run the cells in order. Defaults load `data/processed/sciar_refereed_v1/papers.parquet`. A CUDA GPU is recommended; the notebook also supports CPU execution. BERT weights and its tokenizer are downloaded from Hugging Face on first use.

For a smoke run, use `TRAIN_PAIRS = 64`, `VAL_PAIRS = 32`, `TEST_PAIRS = 32`, and `EPOCHS = 1`. Use a new `OUTPUT_DIR` for each run; existing directories are protected against overwriting.

## Method and outputs

- Split authors 80/10/10 before sampling pairs; no authors or papers cross splits.
- Sample balanced same-/different-author pairs, uniformly by author, with replacement across pairs.
- Independently encode each paper's title and abstract, mean-pool nonpadding token representations, and normalize embeddings.
- Minimize `1 - cosine` for same-author pairs and `max(0, cosine - 0.2)` for different-author pairs.
- Select the best epoch by validation ROC-AUC and the classification threshold by validation balanced accuracy. Evaluate held-out test pairs once.

Outputs under `3_train/outputs/` include `adapter/` (LoRA weights and tokenizer), `author_splits.json`, pair CSVs with bibcodes, `history.csv`, `config.json`, and `metrics.json`. To reuse the adapter, load `AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)` and wrap it with `PeftModel.from_pretrained(base_model, adapter_path)`; use the same tokenization, pooling, normalization, and saved decision threshold.

This is an authorship-similarity baseline. Topic overlap can also drive similarity; random negative pairs do not isolate writing style. First-author identity is a proxy label, not proof of who wrote the text.
