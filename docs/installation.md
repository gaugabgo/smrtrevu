# Installation

## Requirements

- Python 3.11 or higher
- pip

## Install from source

```bash
git clone https://github.com/your-org/revu.git
cd revu
pip install -e .
```

The `-e` flag installs in editable mode, which is recommended while the library is under active development.

## spaCy language model

Revu uses spaCy for abstract preprocessing. After installing the package, download the English model:

```bash
python -m spacy download en_core_web_sm
```

This only needs to be done once per environment.

## Virtual environment (recommended)

It is strongly recommended to install Revu in an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

## Verify installation

```bash
revu --help
```

You should see the top-level command listing. If the command is not found, make sure your virtual environment is activated.

## Key dependencies

The following packages are installed automatically via `setup.py`:

| Package | Purpose |
|---|---|
| `click` | CLI framework |
| `pandas`, `numpy` | Data handling |
| `spacy` | Text preprocessing (NLP) |
| `sentence-transformers` | Document embeddings |
| `bertopic` | Topic modeling |
| `umap-learn`, `hdbscan` | Dimensionality reduction and clustering |
| `networkx`, `python-louvain` | Graph and community analysis |
| `plotly`, `matplotlib`, `seaborn`, `datamapplot` | Visualization |
| `nbib`, `rispy` | Citation file parsing |
| `geopandas`, `pycountry` | Geographic analysis |

## Embedding models

By default, Revu uses `all-MiniLM-L6-v2` from Sentence Transformers. This model is downloaded automatically on first use from HuggingFace. An internet connection is required the first time; subsequent runs use the cached model.

To use a different model, pass `--model_name` to `revu model embed` or `revu model fit`.

## Notes

- The `revu model embed` step can be memory-intensive for large corpora. Running it on a machine with at least 16 GB RAM is recommended for datasets above 50,000 abstracts.
- The `revu preprocess-texts` command supports multiprocessing. By default it uses all available CPU cores minus one. Adjust with `--n-workers`.
