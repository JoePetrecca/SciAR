# BERT LoRA training

`train_lora.ipynb` learns author-comparison embeddings from paper titles and abstracts using a shared BERT encoder and cosine contrastive loss (margin `0.2`). LoRA updates only attention query/value adapters.

## Run

From the repository root, install dependencies in your notebook environment:

```bash
python -m pip install -r 3_train/requirements.txt
python -m jupyterlab 3_train/train_lora.ipynb
```

Select that environment's kernel. Review **Configuration**, then run the cells in order. The defaults read the first 10,000 papers from `data/processed/sciar_refereed_v0/papers.parquet` for a small test run. BERT weights and its tokenizer are downloaded from Hugging Face on first use.

The small run uses 64 train, 32 validation, and 32 test pairs, with one epoch and a maximum of 128 tokens per paper. On CUDA it uses one GPU with float16 mixed precision and gradient scaling; CPU uses full precision. Make sure the selected kernel's PyTorch build can use your GPU (`torch.cuda.is_available()` should return `True`). Set `MAX_PAPERS = None` to load the entire file for a larger run and raise the pair counts and epochs. Use a new `OUTPUT_DIR` for each run; existing directories are protected against overwriting.

## Method and outputs

- Load the tagged functions from `2_Data/data_reader.ipynb` without running its demo, then split authors 80/10/10 before sampling pairs; no authors or papers cross splits.
- Sample balanced same-/different-author pairs, uniformly by author, with replacement across pairs.
- Independently encode each paper's title and abstract as a single text, mean-pool nonpadding token representations, and normalize embeddings.
- Minimize `1 - cosine` for same-author pairs and `max(0, cosine - 0.2)` for different-author pairs.
- Select the best epoch by validation ROC-AUC and the classification threshold by validation balanced accuracy. Evaluate held-out test pairs once.

The reader's pair labels are `1` (same author) and `0` (different authors); only the loss converts `0` to `-1`. Authors with one paper can appear in different-author pairs. Outputs under `3_train/outputs/` include `adapter/` (LoRA weights and tokenizer), `author_splits.json`, pair CSVs with bibcodes and 0/1 labels, `history.csv`, `config.json`, and `metrics.json`. To reuse the adapter, load `AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)` and wrap it with `PeftModel.from_pretrained(base_model, adapter_path)`; use the same tokenization, pooling, normalization, and saved decision threshold.

This is an authorship-similarity baseline. Topic overlap can also drive similarity; random negative pairs do not isolate writing style. First-author identity is a proxy label, not proof of who wrote the text.
