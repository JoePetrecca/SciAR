# SciAR

Scientific authorship representation and verification using paper titles and abstracts. The goal is to test whether text representations capture author identity beyond similarities in research topic, with applications to author name disambiguation.

## Current progress

The repository currently contains data exploration and dataset-building notebooks. Model training and evaluation are the next stages.

| Notebook | Purpose |
| --- | --- |
| `data_analysis.ipynb` | Explore corpus size, first-author ORCID coverage, papers per author, and title/abstract lengths. |
| `build_dataset.ipynb` | Filter, validate, deduplicate, and export a fixed working dataset. |

## Data

The source is a local ADS/SciX snapshot stored as year-partitioned Parquet files under `NASA_ADS_UNIFIED`. The working range is **2014–2024 inclusive**, across scientific fields.

The builder keeps records with:

- The exact `REFEREED` property.
- A nonblank title, abstract, and bibcode.
- First-author metadata matching the first entry in the author list.
- One consistent, valid-format first-author ORCID across the available sources.

ORCID lists are treated as positional when their lengths match the author list. Conflicting candidate IDs are excluded, including conflicts across eligible duplicate records. The placeholder `0000-0001-2345-6789` is excluded as a first-author identity.

The output contains one record per bibcode and preserves the first nonblank title and original abstract. `MIN_PAPERS_PER_AUTHOR` controls whether singleton authors are retained. Singleton authors can contribute different-author pairs, but cannot supply same-author pairs using distinct papers.

ORCID format is checked, but identifiers are not externally verified. First-author identity does not establish who wrote the text.

## Running the notebooks

Use the `SciARvenv` Jupyter kernel on Moria. The notebooks use DuckDB, pandas, NumPy, Matplotlib, and IPython.

Run the explorer to inspect the data. In the builder, set `DATA_DIR`, `OUTPUT_DIR`, `YEARS`, and the minimum paper count, then run all cells in order. The current resource settings are 64 threads and a 256 GB DuckDB memory limit.

The builder exports:

- `papers.parquet`: the filtered paper table.
- `manifest.json`: the selection rules, counts, and build metadata.

Existing output directories are protected against overwriting. Use a new version directory when changing the selection rules. Keep source data, generated datasets, and DuckDB temporary files out of Git.

## Next steps

Assign authors to disjoint training, validation, and test sets before generating pairs. Balance same-author and different-author examples within each split. Only title and abstract text will enter the text encoder; names and ORCIDs are metadata and labels.
