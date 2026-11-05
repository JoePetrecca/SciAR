# Data reader

Install `pandas`, `numpy`, and `pyarrow`, open `2_Data/data_reader.ipynb` in Jupyter, and run the cells in order. Start Jupyter from the repository root or `2_Data/` so the data path resolves correctly.

The demo reads the first 10,000 papers from the existing `data/processed/sciar_refereed_v0/papers.parquet` file (the version that includes single-paper authors). It splits authors 80/10/10, makes 10 balanced pairs per split, prints three examples, and checks labels and author overlap. It does not download or save data.

To make more pairs, change `n_pairs` in the demo cell. Set `max_papers` for a small run or leave it as `None` to read the whole file. `n_pairs` must be even; pairs may repeat.

`text_a` and `text_b` are **X**: each paper's title and abstract. `label` is **Y**: `1` means the two distinct papers share a first-author ORCID; `0` means different first-author ORCIDs. `bibcode_a` and `bibcode_b` track the papers and are not model inputs. ORCIDs are only used for labels and splitting.
