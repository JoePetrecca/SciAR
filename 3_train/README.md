# BERT LoRA training

`train_lora.ipynb` learns author-comparison embeddings from paper titles and abstracts using a shared BERT encoder and cosine contrastive loss (margin `0.2`). LoRA updates only attention query/value adapters. Set `PAIR_READER` to `random` for the original sampler or `specter` for high-topic-similarity pairs.

## Run

From the repository root, install dependencies in your notebook environment:

```bash
python -m pip install -r 3_train/requirements.txt
python -m jupyterlab 3_train/train_lora.ipynb
```

Select that environment's kernel. Review **Configuration**, then run the cells in order. The current configuration uses `sciar_refereed_v1` and the full dataset. BERT weights and its tokenizer are downloaded from Hugging Face on first use.

On CUDA training uses one GPU with float16 mixed precision and gradient scaling; CPU uses full precision. Make sure the selected kernel's PyTorch build can use your GPU (`torch.cuda.is_available()` should return `True`). Set `MAX_PAPERS` to a number for a small trial, or `None` for the full dataset. Use a new `OUTPUT_DIR` for each run; existing directories are protected against overwriting, including after an interrupted run.

### SPECTER pair reader

The SPECTER reader downloads `allenai/specter`, encodes each title and abstract using CLS pooling, and caches normalized float32 embeddings in `data/cache/specter/`. It uses all visible CUDA devices by default, one model worker per GPU; CPU is supported in float32. Configure `SPECTER_DEVICES` to select devices and `SPECTER_PRECISION` to choose `float16` (default on CUDA) or `float32`. FP16 model inference can change similarity scores and rankings slightly; similarity search itself uses float32. The cache key includes model name, precision, input text and order, and token limit, so full-precision and FP16 caches are separate. Completed 8,192-paper shards are saved atomically and reused after interruption if the inputs, precision, and shard size match. Do not launch concurrent jobs against the same cache. Model downloads require access to Hugging Face.

Before splitting authors, SPECTER mode groups records with identical normalized titles (at least 20 characters) or abstracts (at least 100 characters). It keeps one record per same-author group and removes entire groups with conflicting author IDs. This prevents detected duplicate works from crossing splits, but is not exhaustive near-duplicate detection. Counts are saved in `dedup_stats.json`. Positive pairs are seeded random distinct papers from the same author; pair candidates with title similarity >= 0.90 or abstract similarity >= 0.95 are rejected. Each positive anchor is paired with a different-author paper whose SPECTER similarity is closest to that positive score, subject to the same near-duplicate guard and shared limits of 2 appearances per paper and 10 pairs per author. `match_id` links the positive and negative rows in the CSV. Similarity distributions and matched-score gaps are printed. Pairs have balanced labels; near-duplicate detection is heuristic and cross-split near-duplicates may remain.

The full v1 dataset contains hundreds of thousands of papers. Cached embeddings for the original unfiltered input are reused, then subset after filtering. Matching compares each positive anchor against eligible papers in its split; it uses one GPU and `SPECTER_MATCH_BATCH_SIZE` controls comparison memory. Start with a smaller `MAX_PAPERS` and pair counts to check timing and memory (a smaller input requires its own embedding cache). Training reports both pretrained and fine-tuned held-out test AUC; model and threshold selection remain validation-only. Report matched-score gaps when interpreting results, since exact matching is not always possible.

## Method and outputs

- Load the tagged functions from `2_Data/data_reader.ipynb` without running its demo, then split authors 80/10/10 before sampling pairs; no authors or papers cross splits.
- Random mode samples balanced pairs uniformly by author with replacement; SPECTER mode uses distinct positives with topic-matched negatives and the diversity limits described above.
- Independently encode each paper's title and abstract as a single text, mean-pool nonpadding token representations, and normalize embeddings.
- Minimize `1 - cosine` for same-author pairs and `max(0, cosine - 0.2)` for different-author pairs.
- Select the best epoch by validation ROC-AUC and the classification threshold by validation balanced accuracy. Evaluate held-out test pairs once.

The reader's pair labels are `1` (same author) and `0` (different authors); only the loss converts `0` to `-1`. Authors with one paper can appear in different-author pairs. Outputs under `3_train/outputs/` include `adapter/` (LoRA weights and tokenizer), `author_splits.json`, pair CSVs with bibcodes and 0/1 labels, `history.csv`, `config.json`, and `metrics.json`. To reuse the adapter, load `AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)` and wrap it with `PeftModel.from_pretrained(base_model, adapter_path)`; use the same tokenization, pooling, normalization, and saved decision threshold.

This is an authorship-similarity baseline. Topic overlap can also drive similarity; random negative pairs do not isolate writing style. First-author identity is a proxy label, not proof of who wrote the text.
