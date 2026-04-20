# Revu

Revu is a command-line toolkit for systematic literature review. It covers the full pipeline from raw citation exports to interactive topic maps and bibliometric visualizations — designed for researchers who want reproducible, scriptable workflows without writing custom code.

## Capabilities

**Corpus management** — Option to retrieve article title, abstract, and metadata directly from OpenAlex or to parse citation exports from PubMed (NBIB) and other databases (RIS). Normalize and merge records from multiple sources, and deduplicate by DOI and title.

**Topic modeling** — Preprocess abstracts, compute document embeddings, and fit a BERTopic model to identify research themes. The modular pipeline lets you cache embeddings and reuse them across modeling runs. Evaluate the performance of different hyperparameters when fitting a BERTopic model to select the best parameters.

**Bibliometric and co-authorship analysis** — Analyze citation networks, topic influence, and temporal trends. Map author collaboration patterns and generate publication-quality visualizations.

## Requirements

- Python 3.11+
- spaCy English model: `python -m spacy download en_core_web_sm`

See [Installation](docs/installation.md) for full setup instructions.

## Installation

```bash
git clone https://github.com/your-org/revu.git
cd revu
pip install -e .
python -m spacy download en_core_web_sm
```

## Quick Start

The following example goes from raw citation files to a topic model in five steps:

```bash
# 1. Parse citation exports
revu parse nbib data/raw/ data/parsed/ citations.csv

# 2. Merge and deduplicate
revu merge -i "data/parsed/*.csv" -o data/merged.csv
revu deduplicate -i data/merged.csv -o data/dedup.csv -l data/dedup.log

# 3. Preprocess abstracts
revu preprocess-texts --source-type abstract --input-path data/dedup.csv

# 4. Fit a topic model
revu model embed --input_csv data/processed_dedup.csv --output_path data/embeddings.npy
revu model fit --input_csv data/processed_dedup.csv --embeddings_path data/embeddings.npy --output_csv data/topics.csv

# 5. Visualize
revu model reduce --embeddings_path data/embeddings.npy --output_path data/embeddings_2d.npy
revu model visualize --input_csv data/topics.csv --model_path data/model_output/ --embeddings_2d_path data/embeddings_2d.npy
```

## Documentation

| Document | Description |
|---|---|
| [Installation](docs/installation.md) | Full setup, dependencies, and environment notes |
| [Quick Start](docs/quickstart.md) | End-to-end walkthrough with a real example |
| [CLI Reference](docs/cli-reference.md) | All commands, options, and flags |
| [Citation Pipeline](docs/workflows/citation-pipeline.md) | Parsing, merging, and deduplication |
| [Topic Modeling](docs/workflows/topic-modeling.md) | Embedding, fitting, and visualization |
| [Bibliometric Analysis](docs/workflows/bibliometric-analysis.md) | Citation networks, influence, and trends |
| [Co-authorship Analysis](docs/workflows/coauthorship-analysis.md) | Author networks and community detection |

## License

MIT
