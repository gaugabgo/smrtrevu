# Co-authorship Analysis

This document covers the co-authorship analysis pipeline: building author collaboration networks, detecting communities, and generating visualizations.

> **Note:** The co-authorship analysis and its associated visualizations are under active development. Core analysis outputs are stable; visualization commands may change in future releases.

## Prerequisites

- `metadata_validated.csv` from `revu extract-metadata` with author and affiliation data
- `topics.csv` from `revu model fit`
- `embeddings_2d.npy` from `revu model reduce` (required for DataMapPlot-based visualizations)

## Overview

```
metadata_validated.csv + topics.csv
      |
revu biblio coauthorship-analysis    → author_paper_edges.csv
      |                                 coauthorship_edges.csv
      |                                 author_communities.csv
      |                                 author_network_metrics.csv
      |                                 author_network.graphml
      |                                 topic_N/ (per-topic, if --per_topic)
      |
  Visualization commands (see below)
```

## Step 1: Run co-authorship analysis

```bash
revu biblio coauthorship-analysis \
  --metadata_csv data/metadata_validated.csv \
  --topic_csv data/topics.csv \
  --output_dir data/coauthorship_results/
```

To also compute per-topic author networks:

```bash
revu biblio coauthorship-analysis \
  --metadata_csv data/metadata_validated.csv \
  --topic_csv data/topics.csv \
  --output_dir data/coauthorship_results/ \
  --per_topic \
  --min_works_per_topic 30
```

The `--min_works_per_topic` threshold skips topics with too few publications to produce meaningful networks.

**Outputs:**

| File | Contents |
|---|---|
| `author_paper_edges.csv` | One row per author-paper pair: author name, affiliation, country, year |
| `coauthorship_edges.csv` | Author pairs with co-authorship weight (number of shared papers) |
| `author_communities.csv` | Author to community assignment (Louvain detection) |
| `author_network_metrics.csv` | Per-author: degree, strength, betweenness centrality |
| `author_network.graphml` | Full author network for import into Gephi or Cytoscape |
| `topic_N/` | Per-topic versions of the above (with `--per_topic`) |

## Step 2: Visualizations

### Author network map

An interactive DataMapPlot where each document is a point positioned by its 2D embedding, and authors are overlaid based on their publications:

```bash
revu biblio visualize-author-network \
  --embeddings data/embeddings_2d.npy \
  --modeled data/topics.csv \
  --metadata data/metadata_validated.csv \
  --communities data/coauthorship_results/author_communities.csv \
  --metrics data/coauthorship_results/author_network_metrics.csv \
  --topic-info data/model_output/topic_info.csv \
  --min-papers 5 \
  --output data/visualizations/author_network.html
```

Use `--static` to produce a PNG for publication instead of an interactive HTML file.

### Hub-spoke network by topic

A hub-spoke diagram showing the most prolific authors in each topic, colored by topic:

```bash
revu biblio visualize-coauth-topic \
  --coauthorship-dir data/coauthorship_results/ \
  --top-n-topics 20 \
  --top-k-per-topic 10 \
  --topic-info-file data/model_output/topic_info.csv \
  --output data/visualizations/coauth_hub_topic.png
```

### Hub-spoke network by global community

A hub-spoke diagram showing the global author community structure, colored by community:

```bash
revu biblio visualize-coauth-global \
  --coauthorship-dir data/coauthorship_results/ \
  --top-n 100 \
  --output data/visualizations/coauth_hub_global.png
```

### Topic map with co-authorship overlay

An interactive topic DataMapPlot with a toggleable co-authorship community overlay. Requires an aggregation step first:

```bash
# 1. Aggregate per-topic community centroids
revu biblio agg-within-topic-community-nodes \
  --coauthorship-dir data/coauthorship_results/ \
  --modeled data/topics.csv \
  --embeddings data/embeddings_2d.npy \
  --output data/within_topic_community_nodes.csv

# 2. Generate the overlay visualization
revu biblio visualize-dmp-coauth-overlay \
  --modeled data/topics.csv \
  --embeddings data/embeddings_2d.npy \
  --community-nodes data/within_topic_community_nodes.csv \
  --topic-info data/model_output/topic_info.csv \
  --metadata data/metadata_validated.csv \
  --output data/visualizations/topic_dmp_coauth_overlay.html
```

### Topic-community network

An interactive network graph showing the relationship between research topics and author communities:

```bash
# 1. Compute global community-topic affinity
revu biblio agg-global-community-topic-affinity \
  --global-communities data/coauthorship_results/author_communities.csv \
  --global-paper-edges data/coauthorship_results/author_paper_edges.csv \
  --modeled data/topics.csv \
  --embeddings data/embeddings_2d.npy \
  --output-dir data/coauthorship_results/global_community_topic/

# 2. Visualize
revu biblio visualize-topic-community-network \
  --affinity data/coauthorship_results/global_community_topic/topic_community_affinity.csv \
  --edges data/coauthorship_results/global_community_topic/topic_topic_edges.csv \
  --output data/visualizations/topic_community_network.html \
  --top-n-edges 200
```

## Notes

- Author name disambiguation is not currently performed. Authors with identical names across institutions are treated as the same author.
- The Louvain community detection algorithm is non-deterministic. Use `--seed` on visualization commands for reproducible layouts.
- The `author_network.graphml` file can be imported into Gephi for more detailed network exploration and layout customization.
