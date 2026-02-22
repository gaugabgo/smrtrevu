# Bibliographic Visualization Filters

The `revu biblio visualize` command supports comprehensive filtering to help you focus on the most interpretable and interesting topics when you have many topics in your analysis.

## Topic Names

**IMPORTANT**: To display topic names instead of just numbers, use the `--topic_info_file` option to provide a CSV file with topic names.

The topic info file should have columns for topic ID and name. The visualizer will automatically detect column names like:
- Topic ID: `Topic`, `topic`, `topic_id`, `TopicID`
- Topic Name: `Name`, `name`, `topic_name`, `TopicName`, `Representation`

Example:
```bash
revu biblio visualize \
  --data_dir data/results \
  --topic_info_file causalinference_modeled_03Jan2026_topic_info.csv \
  --top_n_prevalent 15
```

Topic labels in visualizations will show as: `{TopicID}: {TopicName}` (e.g., "42: Machine Learning Applications")

## Filter Categories

### 1. Prevalence Filters
Filter topics by how common they are (number of works).

**Options:**
- `--top_n_prevalent N` - Keep only the top N most prevalent topics
- `--min_works N` - Keep only topics with at least N works

**Example:**
```bash
# Show only the 20 most prevalent topics
revu biblio visualize \
  --data_dir data/results \
  --top_n_prevalent 20

# Show only topics with at least 50 works
revu biblio visualize \
  --data_dir data/results \
  --min_works 50
```

### 2. Centrality Filters
Filter topics by their network centrality (importance in citation network).

**Options:**
- `--top_n_central N` - Keep only the top N most central topics
- `--bottom_n_central N` - Keep only the bottom N least central topics
- `--centrality_metric METRIC` - Which centrality metric to use (default: `mean_pagerank`)
  - Available metrics: `mean_pagerank`, `mean_in_degree_centrality`, `mean_out_degree_centrality`, `mean_betweenness`

**Example:**
```bash
# Show top 15 most central topics by PageRank
revu biblio visualize \
  --data_dir data/results \
  --viz_type centrality \
  --top_n_central 15

# Show top 10 most central by betweenness (bridge topics)
revu biblio visualize \
  --data_dir data/results \
  --viz_type centrality \
  --top_n_central 10 \
  --centrality_metric mean_betweenness
```

### 3. Homophily Filters
Filter topics by their citation homophily (tendency to cite within topic).

**Options:**
- `--top_n_homophilic N` - Keep only the top N most homophilic topics
- `--bottom_n_homophilic N` - Keep only the bottom N least homophilic topics
- `--min_excess_homophily VALUE` - Keep only topics with excess homophily ≥ VALUE

**Example:**
```bash
# Show most homophilic topics (high internal citation)
revu biblio visualize \
  --data_dir data/results \
  --viz_type homophily \
  --top_n_homophilic 12

# Show least homophilic topics (high cross-topic citation)
revu biblio visualize \
  --data_dir data/results \
  --viz_type homophily \
  --bottom_n_homophilic 10

# Show topics with significant excess homophily
revu biblio visualize \
  --data_dir data/results \
  --viz_type homophily \
  --min_excess_homophily 0.1
```

### 4. Influence Filters
Filter topics by their influence (external citations, impact).

**Options:**
- `--top_n_influential N` - Keep only the top N most influential topics
- `--bottom_n_influential N` - Keep only the bottom N least influential topics
- `--influence_metric METRIC` - Which influence metric to use (default: `total_external_citations`)
  - Available metrics: `total_external_citations`, `citation_balance`, `h_index_proxy`

**Example:**
```bash
# Show most influential topics by external citations
revu biblio visualize \
  --data_dir data/results \
  --viz_type influence \
  --top_n_influential 15

# Show topics with best citation balance
revu biblio visualize \
  --data_dir data/results \
  --viz_type influence \
  --top_n_influential 10 \
  --influence_metric citation_balance

# Show topics with highest h-index proxy
revu biblio visualize \
  --data_dir data/results \
  --viz_type influence \
  --top_n_influential 12 \
  --influence_metric h_index_proxy
```

### 5. Trend Filters
Filter topics by their temporal trends (growth/decline over time).

**Options:**
- `--top_n_growing N` - Keep only the top N fastest growing topics
- `--bottom_n_declining N` - Keep only the bottom N fastest declining topics
- `--growing_only` - Keep only topics with positive trend slope
- `--declining_only` - Keep only topics with negative trend slope

**Example:**
```bash
# Show fastest growing topics
revu biblio visualize \
  --data_dir data/results \
  --viz_type trends \
  --top_n_growing 15

# Show fastest declining topics
revu biblio visualize \
  --data_dir data/results \
  --viz_type trends \
  --bottom_n_declining 10

# Show all growing topics with substantial presence
revu biblio visualize \
  --data_dir data/results \
  --viz_type trends \
  --growing_only \
  --min_works 30

# Compare fastest growing vs fastest declining
revu biblio visualize \
  --data_dir data/results \
  --viz_type trends \
  --top_n_growing 10 \
  --bottom_n_declining 10
```

## Combining Filters

You can combine multiple filters to create highly focused visualizations:

```bash
# Show most central topics among the prevalent ones
revu biblio visualize \
  --data_dir data/results \
  --top_n_prevalent 50 \
  --top_n_central 15

# Show influential growing topics
revu biblio visualize \
  --data_dir data/results \
  --growing_only \
  --top_n_influential 12 \
  --min_works 25

# Apply the same filters across all visualization types
revu biblio visualize \
  --data_dir data/results \
  --viz_type all \
  --top_n_prevalent 20 \
  --min_works 40
```

## Filter Application Logic

- **Prevalence filters** (`--top_n_prevalent`, `--min_works`) are applied to ALL visualization types
- **Type-specific filters** are only applied to their corresponding visualization:
  - Homophily filters → homophily visualization
  - Centrality filters → centrality visualization
  - Influence filters → influence visualization
  - Trend filters → trends visualization
- Filters are applied **sequentially** (first prevalence, then type-specific)
- When using `--viz_type all`, each visualization gets filtered independently

## Best Practices

1. **Start with prevalence**: Use `--top_n_prevalent` to get an overview
   ```bash
   revu biblio visualize --data_dir data/results --top_n_prevalent 20
   ```

2. **Focus on extremes**: Use `top_n` and `bottom_n` together to highlight contrasts
   ```bash
   revu biblio visualize --data_dir data/results --viz_type trends \
     --top_n_growing 8 --bottom_n_declining 8
   ```

3. **Combine for insights**: Mix filters to answer specific questions
   ```bash
   # "What are the emerging influential topics?"
   revu biblio visualize --data_dir data/results \
     --growing_only --top_n_influential 10 --min_works 20
   ```

4. **Iterate**: Start broad, then narrow down
   - First pass: `--top_n_prevalent 30`
   - Second pass: `--top_n_prevalent 30 --top_n_central 15`
   - Final pass: `--top_n_prevalent 30 --top_n_central 15 --growing_only`

## Output

When filters are applied, the command will print:
- Which filters are being applied
- How many topics remain after filtering
- The total number of topics being visualized

Example output:
```
============================================================
Generating Trends Visualization...
============================================================
Applying prevalence filters...
  Filtered to top 20 most prevalent topics
Applying trend filters...
  Filtered to growing topics only (12 topics)
Visualizing 12 topics
✓ Trends visualization saved
```
