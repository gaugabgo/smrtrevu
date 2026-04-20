# Topic Modeling

This document covers the topic modeling pipeline: preprocessing abstracts, computing embeddings, fitting a BERTopic model, and generating visualizations.

## Overview

```
dedup.csv
    |
revu preprocess-texts
    |
processed_dedup.csv
    |
revu model embed         → embeddings.npy
    |
revu model fit           → topics.csv, topic_info.csv, bertopic_model/
    |
revu model reduce        → embeddings_2d.npy
    |
revu model visualize     → HTML, PNG, CSV
```

The pipeline is intentionally modular. Embeddings are computed once and saved to disk, so you can experiment with different clustering parameters (`--min_cluster_size`, `--n_neighbors`) without recomputing embeddings each time.

## Step 1: Preprocess abstracts

```bash
revu preprocess-texts \
  --source-type abstract \
  --input-path data/dedup.csv
```

Output: `data/processed_dedup.csv` with a `processed_text` column containing cleaned and lemmatized text.

**For large datasets**, enable checkpointing so the job can be safely interrupted and resumed:

```bash
revu preprocess-texts \
  --source-type abstract \
  --input-path data/dedup.csv \
  --checkpoint-dir data/checkpoints/ \
  --n-workers 8
```

The `--threshold` parameter (default `0.95`) controls ubiquitous word removal: any token that appears in more than 95% of documents is treated as uninformative and removed. Lower this value to remove more words; raise it to be more conservative.

## Step 2: Compute embeddings

```bash
revu model embed \
  --input_csv data/processed_dedup.csv \
  --output_path data/embeddings.npy
```

This step uses `all-MiniLM-L6-v2` by default, a well-balanced model for semantic similarity. For domain-specific corpora (e.g. biomedical literature), a domain-adapted model such as `allenai-specter` may improve topic quality:

```bash
revu model embed \
  --input_csv data/processed_dedup.csv \
  --output_path data/embeddings.npy \
  --model_name allenai-specter
```

Models are downloaded automatically from HuggingFace on first use.

**Note:** The row order in `embeddings.npy` must match the row order in the input CSV. Do not sort or filter the CSV after computing embeddings.

## Step 3: Fit the topic model

```bash
revu model fit \
  --input_csv data/processed_dedup.csv \
  --embeddings_path data/embeddings.npy \
  --output_csv data/topics.csv \
  --output_dir data/model_output/
```

**Key tuning parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--min_cluster_size` | `30` | Minimum documents per topic. Increase for fewer, broader topics; decrease for more granular topics. |
| `--n_neighbors` | `15` | UMAP neighborhood size. Higher values produce more global structure. |
| `--n_components` | `5` | Dimensionality before clustering. Usually does not need tuning. |
| `--cluster_selection_method` | `eom` | `eom` (excess of mass) produces more varied cluster sizes; `leaf` produces more uniform sizes. |

**Outlier reduction:** Documents that do not fit any topic are assigned topic `-1`. To reduce the number of outliers, use `--outlier_strategy`:

```bash
revu model fit \
  --input_csv data/processed_dedup.csv \
  --embeddings_path data/embeddings.npy \
  --output_csv data/topics.csv \
  --outlier_strategy embeddings
```

Options: `embeddings` (reassign by embedding proximity), `c-tf-idf` (reassign by term overlap), `probabilities` (use soft cluster probabilities).

**Outputs:**
- `data/topics.csv` — input CSV with `topic` (integer) and `topic_label` columns appended
- `data/model_output/topic_info.csv` — one row per topic: label, document count, representative terms
- `data/model_output/bertopic_model/` — saved model for reuse in visualization

## Step 4: Reduce to 2D

```bash
revu model reduce \
  --embeddings_path data/embeddings.npy \
  --output_path data/embeddings_2d.npy
```

This produces a 2D projection used by the DataMapPlot visualization. The `--random_state` parameter (default `42`) controls reproducibility of the layout.

## Step 5: Visualize

```bash
revu model visualize \
  --input_csv data/topics.csv \
  --model_path data/model_output/bertopic_model/ \
  --embeddings_2d_path data/embeddings_2d.npy \
  --output_dir data/visualizations/
```

Adding metadata enables richer hover tooltips in the interactive map:

```bash
revu model visualize \
  --input_csv data/topics.csv \
  --model_path data/model_output/bertopic_model/ \
  --embeddings_2d_path data/embeddings_2d.npy \
  --metadata_csv data/metadata_validated.csv \
  --output_dir data/visualizations/
```

**Outputs:**

| File | Type | Description |
|---|---|---|
| `topic_map.html` | Interactive HTML | DataMapPlot: each document as a point, colored by topic |
| `topic_barchart.html` | Interactive HTML | Topic frequency bar chart |
| `topic_heatmap.html` | Interactive HTML | Topic similarity heatmap |
| `topic_proportions_*.png` | Static PNG | Topic share over time |
| `topic_proportions_*.csv` | CSV | Raw proportions data |

## Evaluating hyperparameters

Before committing to a final model, use `revu model evaluate` to systematically search the hyperparameter space and rank combinations by topic coherence (c_v score):

```bash
revu model evaluate \
  --input_csv data/processed_dedup.csv \
  --embeddings_path data/embeddings.npy \
  --output_dir data/eval_results/
```

This fits a BERTopic model for every combination of the provided parameter values and records the coherence score, topic count, and outlier rate for each run.

**Default parameter grid:**

| Parameter | Default values |
|---|---|
| `--min-cluster-sizes` | `100 200 300 500` |
| `--n-neighbors-values` | `15 50` |
| `--n-components-values` | `5 15` |

Customize the grid by repeating the option for each value you want to test:

```bash
revu model evaluate \
  --input_csv data/processed_dedup.csv \
  --embeddings_path data/embeddings.npy \
  --output_dir data/eval_results/ \
  --min-cluster-sizes 50 \
  --min-cluster-sizes 100 \
  --min-cluster-sizes 200 \
  --n-neighbors-values 15 \
  --n-neighbors-values 30
```

For very large corpora, use `--sample_size` to evaluate on a random subset:

```bash
revu model evaluate \
  --input_csv data/processed_dedup.csv \
  --embeddings_path data/embeddings.npy \
  --sample_size 5000 \
  --output_dir data/eval_results/
```

**Outputs:**
- `hyperparameter_results_<timestamp>.csv` — full grid results ranked by coherence
- `best_params_<timestamp>.txt` — summary of the top-scoring parameter combination

Once you have identified the best parameters, use them with `revu model fit`.

## Iterating on the model

A typical iteration workflow:

1. Run `model evaluate` to identify the best parameter combination by coherence score.
2. Run `model fit` with those parameters using the same `embeddings.npy`.
3. Inspect `topic_info.csv` to assess topic coverage and label quality.
4. Check the outlier rate: `topic == -1` rows divided by total rows. An outlier rate above 20–30% may indicate `--min_cluster_size` is too large.
5. Once satisfied, run `model reduce` and `model visualize`.

## Linking metadata

Before running bibliometric or co-authorship analysis, validate that your metadata CSV and topic CSV share consistent IDs:

```bash
revu extract-metadata \
  --metadata_deduplicated_csv data/dedup.csv \
  --topic_model_csv data/topics.csv \
  --validated_metadata_csv data/metadata_validated.csv
```

This step filters both files to confirmed 1:1 matches and is required as input for downstream analysis commands.
