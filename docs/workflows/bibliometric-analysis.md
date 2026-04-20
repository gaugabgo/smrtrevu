# Bibliometric Analysis

This document covers the bibliometric analysis pipeline: computing citation network metrics, topic influence and temporal trends, and generating visualizations.

## Prerequisites

- `topics.csv` from `revu model fit`
- `metadata_validated.csv` from `revu extract-metadata` (required for citation network analysis and geographic visualizations; optional for trend analysis)
- Metadata must include a `referenced_works` column (list of cited DOIs/IDs) for citation network features

## Overview

```
topics.csv + metadata_validated.csv
      |
revu biblio analyze        → topic_centrality.csv
      |                       topic_influence.csv
      |                       topic_trends.csv
      |
revu biblio build-network  → citation_network.graphml   (optional)
      |
revu biblio visualize      → PNG / HTML visualizations
```

## Step 1: Run analysis

```bash
revu biblio analyze \
  --topic_csv data/topics.csv \
  --metadata_csv data/metadata_validated.csv \
  --output_dir data/biblio_results/
```

This produces three CSVs that feed into all downstream visualizations:

| File | Contents |
|---|---|
| `topic_centrality.csv` | PageRank, betweenness, and degree centrality aggregated by topic |
| `topic_influence.csv` | External citation counts, h-index proxy, citation patterns by topic |
| `topic_trends.csv` | Temporal publication trends, growth rates, emergence and decline signals |

## Step 2: Build citation network (optional)

Export the citation network as GraphML for use in external tools such as Gephi or Cytoscape:

```bash
revu biblio build-network \
  --topic_csv data/topics.csv \
  --metadata_csv data/metadata_validated.csv \
  --output_path data/citation_network.graphml
```

## Step 3: Visualize

The `biblio visualize` command generates all visualization types by default. Use `--viz_type` to generate a specific type.

```bash
revu biblio visualize \
  --data_dir data/biblio_results/ \
  --output_dir data/visualizations/ \
  --topic_info_file data/model_output/topic_info.csv
```

### Visualization types

#### Centrality network (`--viz_type centrality`)

A network graph where nodes are topics, sized and colored by PageRank centrality. Edges represent citation links between topics.

```bash
revu biblio visualize \
  --data_dir data/biblio_results/ \
  --viz_type centrality \
  --top_n_central 30 \
  --works_file data/biblio_results/topic_works.csv
```

#### Influence chart (`--viz_type influence`)

A ranked bar chart of topics by external citation impact.

```bash
revu biblio visualize \
  --data_dir data/biblio_results/ \
  --viz_type influence \
  --top_n_influential 25 \
  --normalize_by_works
```

The `--normalize_by_works` flag divides citation counts by the number of publications per topic, making topics of different sizes comparable.

#### Temporal trends (`--viz_type trends`)

A span chart showing which topics are growing, declining, or stable over time.

```bash
revu biblio visualize \
  --data_dir data/biblio_results/ \
  --viz_type trends \
  --top_n_each 15 \
  --sort_spans_by trend_slope \
  --color_growing "#2ecc71" \
  --color_declining "#e74c3c"
```

`--top_n_each` selects the top N growing and top N declining topics for a balanced view.

#### Geographic distribution (`--viz_type geo`)

A choropleth map of publication counts by country, derived from author affiliations.

```bash
revu biblio visualize \
  --data_dir data/biblio_results/ \
  --viz_type geo \
  --metadata_csv data/metadata_validated.csv \
  --geo_format interactive
```

Use `--geo_format static` for a PNG suitable for publication, or `--geo_format interactive` for an explorable HTML map.

### Filtering options

All visualization types support prevalence-based filtering to focus on topics with sufficient representation:

```bash
revu biblio visualize \
  --data_dir data/biblio_results/ \
  --min_works 50        # only topics with at least 50 publications
  --top_n_prevalent 40  # further restrict to the 40 most prevalent
```

These filters apply before type-specific filters (e.g. `--top_n_central`).

## Notes

- Topic labels in visualizations are taken from `topic_info.csv` when `--topic_info_file` is provided. Without it, numeric topic IDs are used.
- The geographic visualization requires the metadata to have an `authorships.countries` column in OpenAlex format (a list of country strings per record).
- For very large corpora, the `biblio analyze` step can be slow due to the network construction. The results are cached in `output_dir` and are not recomputed on subsequent visualization runs.
