# Project Step 2: Data

I am using scientific papers from SciX (https://scixplorer.org). I have a local copy of the database in Parquet files, and I'm looking at papers from 2014–2024. I'm using titles and abstracts, with first-author ORCIDs for the author labels.

I've been exploring and filtering the data using DuckDB. Right now I'm keeping refereed papers with nonempty titles and abstracts and a usable first-author ORCID. I'm also removing duplicates and records with conflicting ORCID information.

So far, I've found:

- 718,925 unique first-author ORCIDs
- 1,175,775 papers counted across authors
- 496,052 authors with only one paper
- 222,873 authors with at least two papers
- 27,934 authors with at least five papers
- 3,905 authors with at least ten papers

The mean is 1.64 papers per author, and the median is one.

My first filtered dataset, requiring at least two papers per author, has 679,721 papers from 222,872 authors. I'm considering keeping the singletons as well since they could still be useful for negative pairs.

Each record contains the title, abstract, year, bibcode, and first-author information. One example is *K2 Observations of SN 2018oh Reveal a Two-component Rising Light Curve for a Type Ia Supernova*, with an abstract beginning "We present an exquisite 30 minute cadence Kepler (K2) light curve of the Type Ia supernova..."

I haven't finalized the split yet since I'm still cleaning the data, but it'll likely be 80/10/10 for training, validation, and testing. I'll split by author to avoid overlap between sets, then generate positive and negative pairs.

For representations, I'm planning to start with TF-IDF and pretrained embeddings, probably SciBERT. I also want to try more traditional stylometric features like sentence length, punctuation, and function-word frequencies, either separately or combined with embeddings.

I'll experiment with preprocessing, but initially I'm keeping the text mostly unchanged since some of those features might be useful. I'm particularly interested in how well the different representations separate writing style from topical similarity.

**Data:** https://scixplorer.org

The filtered dataset is stored locally for now.