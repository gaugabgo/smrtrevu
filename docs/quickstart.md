# Quick Start

This guide walks through a complete end-to-end workflow: from raw citation exports to a topic model with visualizations. It assumes you have already completed [installation](installation.md).

## Assumed directory layout

```
project/
├── data/
│   ├── raw/          # Raw .nbib or .ris files exported from your database
│   └── ...
└── ...
```

## Step 1: Parse citation files

Export your citations from PubMed (NBIB format) or a reference manager such as Zotero or Endnote (RIS format), and place the files in `data/raw/`.

```bash
# For PubMed NBIB exports
revu parse nbib data/raw/ data/parsed/ citations.csv

# For RIS exports
revu parse ris data/raw/ data/parsed/ citations.csv
```

This produces `data/parsed/citations.csv` with columns: `id`, `doi`, `title`, `language`, `abstract`, `journal`, `year`, `authors`.

If your RIS files came from EBSCO, normalize date formats first:

```bash
revu preprocess data/raw/ data/raw_normalized/
revu parse ris data/raw_normalized/ data/parsed/ citations.csv
```

## Step 2: Merge and deduplicate

If you have exports from multiple databases, merge them before deduplication:

```bash
revu merge -i "data/parsed/*.csv" -o data/merged.csv
```

Then deduplicate by DOI and title:

```bash
revu deduplicate -i data/merged.csv -o data/dedup.csv -l data/dedup.log
```

The log file records which records were removed and why. Only English-language records with abstracts are retained.

## Step 3: Preprocess abstracts

Clean and normalize the abstract text for topic modeling:

```bash
revu preprocess-texts \
  --source-type abstract \
  --input-path data/dedup.csv
```

This produces `data/processed_dedup.csv` with a `processed_text` column. For large datasets, use checkpointing so the job can be resumed if interrupted:

```bash
revu preprocess-texts \
  --source-type abstract \
  --input-path data/dedup.csv \
  --checkpoint-dir data/checkpoints/
```

## Step 4: Compute embeddings

Compute document embeddings and save them to disk. Separating this step lets you reuse the embeddings across multiple modeling runs without recomputing them.

```bash
revu model embed \
  --input_csv data/processed_dedup.csv \
  --output_path data/embeddings.npy
```

## Step 5: Fit the topic model

```bash
revu model fit \
  --input_csv data/processed_dedup.csv \
  --embeddings_path data/embeddings.npy \
  --output_csv data/topics.csv \
  --output_dir data/model_output/
```

This produces:
- `data/topics.csv` — the input CSV with `topic` and `topic_label` columns added
- `data/model_output/topic_info.csv` — one row per topic with label, size, and representative words
- `data/model_output/bertopic_model/` — the saved model directory

Documents that do not fit any topic are assigned topic `-1` (outliers).

## Step 6: Reduce embeddings for visualization

Visualizations require 2D coordinates. Reduce the high-dimensional embeddings:

```bash
revu model reduce \
  --embeddings_path data/embeddings.npy \
  --output_path data/embeddings_2d.npy
```

## Step 7: Generate visualizations

```bash
revu model visualize \
  --input_csv data/topics.csv \
  --model_path data/model_output/bertopic_model/ \
  --embeddings_2d_path data/embeddings_2d.npy \
  --output_dir data/visualizations/
```

Output includes:
- An interactive topic map (DataMapPlot, HTML)
- A topic frequency bar chart (HTML)
- A topic similarity heatmap (HTML)
- Topic proportions over time (CSV + PNG)

## Next steps

- To tune the model (cluster size, outlier handling), see [Topic Modeling](workflows/topic-modeling.md).
- To analyze citation networks and author collaboration, see [Bibliometric Analysis](workflows/bibliometric-analysis.md) and [Co-authorship Analysis](workflows/coauthorship-analysis.md).
- For a full command reference, see [CLI Reference](cli-reference.md).
