# CLI Reference

All commands follow the pattern `revu <command> [options]`. Run `revu --help` or `revu <command> --help` for inline help at any level.

---

## Corpus Management

### `revu fetch-openalex`

Retrieve records from the OpenAlex API by keyword search and save to a CSV file. Results include title, abstract, author affiliations, citation counts, and referenced works.

```
revu fetch-openalex --query <query> --output <path> --api-key <key> [options]
```

| Option | Default | Description |
|---|---|---|
| `--query` / `-q` | — | Search query using OpenAlex filter syntax and Boolean operators (required) |
| `--output` / `-o` | — | Output CSV file path (required) |
| `--api-key` | — | OpenAlex API key. Can also be set via `OPENALEX_API_KEY` env var (required) |
| `--types` | `article\|dissertation\|preprint\|book-chapter\|book-section` | Publication types, pipe-separated |
| `--from-date` | `1900-01-01` | Start date (YYYY-MM-DD) |
| `--to-date` | `2099-12-31` | End date (YYYY-MM-DD) |

Output columns: `id`, `doi`, `title`, `publication_year`, `language`, `type`, `cited_by_count`, `referenced_works_count`, `referenced_works`, `primary_topic`, `primary_topic.subfield`, `primary_topic.domain`, `topics`, `authors`, `keywords`, `abstract`.

---

### `revu parse`

Parse a folder of citation files into a single CSV.

```
revu parse <filetype> <input_folder> <output_folder> <output_csv>
```

| Argument | Description |
|---|---|
| `filetype` | `nbib` or `ris` |
| `input_folder` | Directory containing citation files (must exist) |
| `output_folder` | Directory for output files |
| `output_csv` | Filename for the merged output CSV |

Output columns: `id`, `doi`, `au` (authors), `ti` (title), `dp` (date), `jt` (journal), `ab` (abstract).

---

### `revu preprocess`

Normalize date formats in RIS files exported from EBSCO. Run this before `parse` when using EBSCO exports.

```
revu preprocess <input_folder> <output_folder>
```

---

### `revu merge`

Merge multiple CSV files into one.

```
revu merge -i <pattern> -o <output_csv>
```

| Option | Description |
|---|---|
| `-i` / `--input` | Path or glob pattern for input CSVs (e.g. `"data/*.csv"`) |
| `-o` / `--output` | Output CSV path |

---

### `revu deduplicate`

Remove duplicate records by DOI and title. Filters to English-language records and retains records with abstracts when duplicates exist.

```
revu deduplicate -i <input_csv> -o <output_csv> -l <log_file>
```

| Option | Description |
|---|---|
| `-i` / `--input` | Input CSV (must exist) |
| `-o` / `--output` | Deduplicated output CSV |
| `-l` / `--log` | Log file recording removed records |

---

### `revu extract-metadata`

Validate one-to-one ID matching between a deduplicated metadata CSV and a topic model CSV. Outputs only records with a confirmed match.

```
revu extract-metadata \
  --metadata_deduplicated_csv <path> \
  --topic_model_csv <path> \
  --validated_metadata_csv <path> \
  [--log_file <path>]
```

---

## Text Preprocessing

### `revu preprocess-texts`

Preprocess abstract or full-text data. Applies tokenization, lemmatization, stop-word removal, and ubiquitous-word filtering. Supports multiprocessing and checkpointing for large datasets.

```
revu preprocess-texts \
  --source-type <type> \
  --input-path <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--source-type` | — | `abstract` or `fulltext` (required) |
| `--input-path` | — | CSV file (abstract) or directory (fulltext) (required) |
| `--filtering` | `disable` | `enable` or `disable` |
| `--chunk-size` | `1000` | Rows per processing chunk |
| `--n-workers` | CPU count − 1 | Parallel worker count |
| `--checkpoint-dir` | — | Directory for resumable checkpoints |
| `--threshold` | `0.95` | Ubiquitous word removal threshold (0–1) |

Output: `processed_<input_filename>.csv` with a `processed_text` column added.

---

## Topic Modeling

### `revu model embed`

Compute document embeddings and save to disk.

```
revu model embed \
  --input_csv <path> \
  --output_path <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--input_csv` | — | CSV with text to embed (required) |
| `--output_path` | — | Output `.npy` file path (required) |
| `--model_name` | `all-MiniLM-L6-v2` | Sentence Transformers model name |
| `--batch_size` | `32` | Embedding batch size |
| `--text_column` | `processed_text` | Column to embed |

---

### `revu model fit`

Fit a BERTopic model. Accepts pre-computed embeddings to avoid recomputing them across runs.

```
revu model fit \
  --input_csv <path> \
  --output_csv <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--input_csv` | — | CSV with `processed_text` column (required) |
| `--output_csv` | — | Output CSV with topic assignments (required) |
| `--output_dir` | `model_output` | Directory for model files and topic info |
| `--model_name` | `all-MiniLM-L6-v2` | Embedding model (used if no embeddings provided) |
| `--embeddings_path` | — | Pre-computed `.npy` embeddings (optional) |
| `--save_embeddings` | off | Save computed embeddings to `output_dir` |
| `--n_neighbors` | `15` | UMAP n_neighbors |
| `--n_components` | `5` | UMAP n_components |
| `--min_cluster_size` | `30` | HDBSCAN min cluster size |
| `--cluster_selection_method` | `eom` | `eom` or `leaf` |
| `--outlier_strategy` | `none` | `none`, `embeddings`, `c-tf-idf`, or `probabilities` |

Outputs: `<output_csv>`, `<output_dir>/topic_info.csv`, `<output_dir>/bertopic_model/`.

---

### `revu model reduce`

Reduce high-dimensional embeddings to 2D using UMAP, for use with visualizations.

```
revu model reduce \
  --embeddings_path <path> \
  --output_path <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--embeddings_path` | — | High-dimensional `.npy` file (required) |
| `--output_path` | — | Output 2D `.npy` file (required) |
| `--n_neighbors` | `15` | UMAP n_neighbors |
| `--min_dist` | `0.1` | UMAP min_dist |
| `--metric` | `cosine` | UMAP distance metric |
| `--random_state` | `42` | Random seed |

---

### `revu model visualize`

Generate visualizations from a fitted model and 2D embeddings.

```
revu model visualize \
  --input_csv <path> \
  --model_path <path> \
  --embeddings_2d_path <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--input_csv` | — | CSV with topic assignments (required) |
| `--model_path` | — | Saved BERTopic model directory (required) |
| `--embeddings_2d_path` | — | 2D `.npy` embeddings (required) |
| `--output_dir` | `visualizations` | Output directory |
| `--metadata_csv` | — | Metadata CSV for hover text (optional) |
| `--label_mode` | `custom` | `auto` or `custom` |

Outputs: interactive HTML topic map (DataMapPlot), topic bar chart, topic heatmap, topic proportions by year (PNG + CSV).

---

### `revu model evaluate`

Evaluate BERTopic hyperparameters across a grid and rank combinations by topic coherence (c_v score). Fits a model for every combination of the provided parameter values and saves a ranked results CSV and a best-params summary file.

```
revu model evaluate --input_csv <path> [options]
```

| Option | Default | Description |
|---|---|---|
| `--input_csv` | — | CSV with `processed_text` column (required) |
| `--output_dir` | `hyperparameter_results` | Directory for results CSV and best-params file |
| `--embeddings_path` | — | Pre-computed `.npy` embeddings (optional; computed on the fly if omitted) |
| `--model_name` | `all-MiniLM-L6-v2` | Embedding model (used only if no embeddings provided) |
| `--text_column` | `processed_text` | Column containing text to model |
| `--sample_size` | — | Evaluate on a random sample of N documents |
| `--low_memory` | off | Use single-threaded, low-memory UMAP settings |
| `--min-cluster-sizes` | `100 200 300 500` | HDBSCAN `min_cluster_size` values (repeat option for each value) |
| `--n-neighbors-values` | `15 50` | UMAP `n_neighbors` values (repeat option for each value) |
| `--n-components-values` | `5 15` | UMAP `n_components` values (repeat option for each value) |

Outputs: `hyperparameter_results_<timestamp>.csv`, `best_params_<timestamp>.txt`.

---

### `revu model run` _(legacy)_

Single-step topic modeling. Computes embeddings and fits the model in one command. For new workflows, prefer the modular `embed` → `fit` → `reduce` → `visualize` pipeline.

```
revu model run \
  --input_csv <path> \
  --output_csv <path> \
  [options]
```

---

## Bibliometric Analysis

### `revu biblio analyze`

Run bibliometric analysis on a topic model. Computes citation network centrality, topic influence, and temporal trends.

```
revu biblio analyze \
  --topic_csv <path> \
  --output_dir <path> \
  [--metadata_csv <path>] \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--topic_csv` | — | Topic model CSV (required) |
| `--output_dir` | — | Output directory (required) |
| `--metadata_csv` | — | Metadata CSV with citation data (optional) |
| `--topic_column` | `topic` | Topic column name |
| `--id_column` | `id` | ID column name |
| `--window_size` | `10` | Time window for trend analysis |
| `--analysis_type` | all | Specific analysis type (optional) |

Outputs: `topic_centrality.csv`, `topic_influence.csv`, `topic_trends.csv`.

---

### `revu biblio build-network`

Build a directed citation network and export as GraphML (importable to Gephi, Cytoscape, etc.).

```
revu biblio build-network \
  --topic_csv <path> \
  --output_path <path> \
  [--metadata_csv <path>]
```

---

### `revu biblio visualize`

Generate bibliometric visualizations from analysis outputs.

```
revu biblio visualize \
  --data_dir <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--data_dir` | — | Directory with analysis CSVs (required) |
| `--output_dir` | `data_dir` | Visualization output directory |
| `--topic_info_file` | — | Topic names CSV (optional) |
| `--viz_type` | `all` | `all`, `centrality`, `influence`, `trends`, or `geo` |

**Prevalence filters:**

| Option | Description |
|---|---|
| `--top_n_prevalent` | Restrict to top N most prevalent topics |
| `--min_works` | Minimum publications per topic |

**Centrality options:**

| Option | Default | Description |
|---|---|---|
| `--top_n_central` | — | Top N topics by PageRank |
| `--bottom_n_central` | — | Bottom N by centrality |
| `--centrality_metric` | `mean_pagerank` | Metric to sort by |
| `--works_file` | — | Works CSV for paper satellite nodes |

**Influence options:**

| Option | Default | Description |
|---|---|---|
| `--top_n_influential` | — | Top N influential topics |
| `--bottom_n_influential` | — | Bottom N least influential |
| `--influence_metric` | `total_external_citations` | Metric to sort by |
| `--min_citations` | — | Minimum citation threshold |
| `--normalize_by_works` | off | Divide citations by number of works |

**Trend options:**

| Option | Default | Description |
|---|---|---|
| `--top_n_growing` | — | Top N fastest growing topics |
| `--bottom_n_declining` | — | Bottom N fastest declining |
| `--top_n_each` | — | Balanced top N growing + declining |
| `--top_n_spans` | `25` | Max rows in temporal span chart |
| `--sort_spans_by` | `total_works` | `total_works`, `first_year`, `span_length`, or `trend_slope` |
| `--color_growing` | `#2ecc71` | Color for growing topics |
| `--color_declining` | `#e74c3c` | Color for declining topics |

**Geographic options:**

| Option | Default | Description |
|---|---|---|
| `--metadata_csv` | — | Metadata with `authorships.countries` column |
| `--geo_cmap` | `YlOrRd` | Matplotlib colormap |
| `--geo_format` | `static` | `static` (PNG) or `interactive` (HTML) |

---

### `revu biblio coauthorship-analysis`

Analyze co-authorship patterns and author collaboration networks.

```
revu biblio coauthorship-analysis \
  --metadata_csv <path> \
  --output_dir <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--metadata_csv` | — | Metadata with author and affiliation data (required) |
| `--output_dir` | — | Output directory (required) |
| `--topic_csv` | — | Topic model CSV (optional) |
| `--per_topic` | off | Compute per-topic networks |
| `--min_works_per_topic` | — | Minimum publications per topic |
| `--affiliation_mode` | `both` | `author`, `institution`, or `both` |

Outputs: `author_paper_edges.csv`, `coauthorship_edges.csv`, `author_communities.csv`, `author_network_metrics.csv`, `author_network.graphml`. With `--per_topic`, also generates `topic_N/` subdirectories.

---

### `revu biblio visualize-author-network`

Author co-authorship map using DataMapPlot, positioned by UMAP coordinates.

```
revu biblio visualize-author-network [options]
```

| Option | Default | Description |
|---|---|---|
| `--embeddings` | `data/embeddings_2d.npy` | 2D UMAP coordinates |
| `--modeled` | — | Topic model CSV |
| `--metadata` | — | Metadata with author names |
| `--communities` | — | Author communities CSV |
| `--metrics` | — | Author network metrics CSV |
| `--topic-info` | — | Topic info CSV for labels (optional) |
| `--min-papers` | `5` | Exclude authors with fewer than N papers |
| `--top-label-n` | `200` | Top N authors to label |
| `--output` | `author_network_dmp.html` | Output file |
| `--static` | off | Produce static PNG instead of HTML |
| `--title` | `Author Co-authorship Map` | Plot title |

---

### `revu biblio visualize-coauth-topic`

Hub-spoke co-authorship network colored by research topic.

```
revu biblio visualize-coauth-topic \
  --coauthorship-dir <path> \
  [options]
```

| Option | Default | Description |
|---|---|---|
| `--coauthorship-dir` | — | Coauthorship results directory (required) |
| `--top-n-topics` | `20` | Number of topics to include |
| `--top-k-per-topic` | `10` | Top K authors per topic |
| `--label-top-authors` | `3` | Authors per hub to label |
| `--topic-info-file` | — | Topic info CSV (optional) |
| `--output` | `coauth_hub_topic.png` | Output PNG path |
| `--figsize-w` | `22.0` | Figure width (inches) |
| `--figsize-h` | `20.0` | Figure height (inches) |
| `--seed` | `42` | Random seed |

---

### `revu biblio visualize-coauth-global`

Hub-spoke co-authorship network colored by global community.

```
revu biblio visualize-coauth-global [options]
```

| Option | Default | Description |
|---|---|---|
| `--coauthorship-dir` | `data/coauthorship_results` | Coauthorship results directory |
| `--graphml` | — | Author network GraphML (auto-discovered if omitted) |
| `--metrics` | — | Author network metrics CSV |
| `--communities` | — | Author communities CSV |
| `--top-n` | `100` | Top N authors as satellites |
| `--label-top-n` | `0` (all) | Top N authors to label |
| `--output` | `coauth_hub_global.png` | Output PNG |
| `--seed` | `42` | Random seed |

---

### `revu biblio agg-within-topic-community-nodes`

Aggregate per-topic co-authorship community centroids for DataMapPlot overlay.

```
revu biblio agg-within-topic-community-nodes [options]
```

| Option | Default | Description |
|---|---|---|
| `--coauthorship-dir` | `data/coauthorship_results` | Coauthorship results root |
| `--modeled` | — | Topic model CSV |
| `--embeddings` | — | 2D embeddings `.npy` |
| `--output` | `data/within_topic_community_nodes.csv` | Output CSV |
| `--top-k-per-topic` | `5` | Max communities per topic |
| `--min-community-members` | `3` | Minimum authors per community |
| `--top-n-authors` | `5` | Authors to include in labels |

---

### `revu biblio agg-global-community-topic-affinity`

Compute affinity between global author communities and research topics.

```
revu biblio agg-global-community-topic-affinity [options]
```

| Option | Default | Description |
|---|---|---|
| `--global-communities` | `data/coauthorship_results/author_communities.csv` | Author communities CSV |
| `--global-paper-edges` | `data/coauthorship_results/author_paper_edges.csv` | Author-paper edges CSV |
| `--modeled` | — | Topic model CSV |
| `--embeddings` | — | 2D embeddings `.npy` |
| `--topic-info` | — | Topic info CSV (optional) |
| `--output-dir` | `data/coauthorship_results/global_community_topic` | Output directory |
| `--top-n-communities` | `15` | Retain top N communities |
| `--edge-min-weight` | `0.05` | Minimum edge weight threshold |

Outputs: `topic_community_affinity.csv`, `topic_topic_edges.csv`.

---

### `revu biblio visualize-dmp-coauth-overlay`

Interactive topic DataMapPlot with toggleable within-topic co-authorship community overlay.

```
revu biblio visualize-dmp-coauth-overlay [options]
```

| Option | Default | Description |
|---|---|---|
| `--modeled` | — | Topic model CSV |
| `--embeddings` | — | 2D embeddings `.npy` |
| `--community-nodes` | `data/within_topic_community_nodes.csv` | Community nodes CSV |
| `--topic-info` | — | Topic info CSV (optional) |
| `--metadata` | — | Metadata for hover text (optional) |
| `--output` | `data/visualizations/topic_dmp_coauth_overlay.html` | Output HTML |
| `--title` | `Topic Map with Co-authorship Communities` | Plot title |
| `--noise-label` | `Outlier` | Label for outlier documents |
| `--top-k-communities` | — | Keep top K communities per topic |

---

### `revu biblio visualize-topic-community-network`

Interactive Plotly network graph showing topic-community relationships.

```
revu biblio visualize-topic-community-network [options]
```

| Option | Default | Description |
|---|---|---|
| `--affinity` | — | Topic-community affinity CSV |
| `--edges` | — | Topic-topic edges CSV |
| `--output` | `data/visualizations/topic_community_network.html` | Output HTML |
| `--title` | `Topic Community Network` | Figure title |
| `--top-n-edges` | `300` | Keep top N heaviest edges (0 = all) |
| `--min-edge-weight` | `0.0` | Minimum edge weight |
| `--edge-alpha` | `0.35` | Edge opacity (0–1) |
