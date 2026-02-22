# Model Evaluator Guide

The `model_evaluator.py` script helps you find optimal hyperparameters for BERTopic by systematically testing different combinations and evaluating them using coherence scores.

## Overview

The model evaluator:
- Tests multiple hyperparameter combinations in a grid search
- Uses pre-computed embeddings to avoid recomputing them for each combination
- Calculates coherence scores to measure topic quality
- Tracks number of topics and outlier percentages
- Saves detailed results for analysis

## Usage

### Basic Usage with Pre-Computed Embeddings (Recommended)

First, compute embeddings using the CLI:

```bash
# Step 1: Compute embeddings once
revu model embed \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --output_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --batch_size 32
```

Then run the hyperparameter search:

```bash
# Step 2: Run hyperparameter evaluation with pre-computed embeddings
python -m revu.model_evaluator \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --embeddings_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --output_dir data/estimand_review/hyperparameter_results
```

### Without Pre-Computed Embeddings

If you haven't computed embeddings yet, the script will compute them for you:

```bash
python -m revu.model_evaluator \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --output_dir data/estimand_review/hyperparameter_results
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_csv` | Yes | - | Path to CSV with processed text |
| `--embeddings_path` | No | None | Path to pre-computed embeddings (.npy file) |
| `--model_name` | No | "all-MiniLM-L6-v2" | Embedding model name |
| `--output_dir` | No | "hyperparameter_results" | Directory to save results |
| `--text_column` | No | "processed_text" | Column name containing text to model |
| `--sample_size` | No | None | Sample N documents for testing (recommended for large datasets) |
| `--low_memory` | No | False | Use low-memory UMAP settings (slower but less RAM) |

## Hyperparameter Grid

The script tests combinations of these parameters:

```python
{
    "min_cluster_size": [50, 100, 200],    # HDBSCAN minimum cluster size
    "n_neighbors": [10, 30, 50],            # UMAP n_neighbors
    "n_components": [5, 10]                 # UMAP dimensionality
}
```

This creates **18 total combinations** to test.

### Customizing the Grid

To test different parameter values, edit the `param_grid` in the script:

```python
param_grid = ParameterGrid({
    "min_cluster_size": [50, 100, 150, 200, 250],  # Add more values
    "n_neighbors": [10, 20, 30, 50, 70],           # Test more neighbors
    "n_components": [3, 5, 7, 10, 15]              # Try different dimensions
})
```

## Output Files

The script creates the following files in the output directory:

### 1. `hyperparameter_results_YYYYMMDD_HHMMSS.csv`

Complete results table with columns:
- `min_cluster_size`: Tested HDBSCAN parameter
- `n_neighbors`: Tested UMAP parameter
- `n_components`: Tested UMAP parameter
- `coherence`: Coherence score (higher is better)
- `num_topics`: Number of topics found
- `num_outliers`: Number of documents assigned to topic -1
- `outlier_pct`: Percentage of outlier documents
- `status`: Success or failure reason

### 2. `best_params_YYYYMMDD_HHMMSS.txt`

Text file with the best hyperparameters based on coherence score:

```
Best Hyperparameters (by coherence)
==================================================

min_cluster_size: 100
n_neighbors: 30
n_components: 5

Metrics:
  Coherence: 0.5234
  Number of topics: 45
  Outliers: 12543 (3.4%)
```

## Understanding the Results

### Coherence Score
- Measures how semantically similar the words in each topic are
- Range: 0 to 1 (higher is better)
- Typical good values: 0.4 - 0.7
- Values below 0.3 suggest poor topic quality

### Number of Topics
- Too few topics: May be too general
- Too many topics: May be too specific
- Consider your domain knowledge when evaluating

### Outlier Percentage
- Documents that don't fit well into any topic
- 5-15% is typical and acceptable
- Very high (>30%): May need lower `min_cluster_size`
- Very low (<5%): May be forcing documents into poor-fitting topics

## Interpreting Results

### Example Output

```
Top 5 by coherence:
  0.5432 | min_cluster=100, n_neighbors=30, n_components=5 | Topics: 47, Outliers: 8.2%
  0.5398 | min_cluster=100, n_neighbors=50, n_components=5 | Topics: 45, Outliers: 9.1%
  0.5201 | min_cluster=50, n_neighbors=30, n_components=5 | Topics: 68, Outliers: 5.3%
  0.5187 | min_cluster=200, n_neighbors=30, n_components=5 | Topics: 31, Outliers: 12.4%
  0.5134 | min_cluster=100, n_neighbors=30, n_components=10 | Topics: 43, Outliers: 8.5%
```

**What to look for:**
1. **Highest coherence** - Better topic quality
2. **Reasonable topic count** - Not too many, not too few
3. **Acceptable outlier rate** - 5-15% is good
4. **Domain fit** - Does the number of topics make sense for your data?

### Parameter Effects

**min_cluster_size:**
- **Higher values** → Fewer topics, more outliers, potentially higher coherence
- **Lower values** → More topics, fewer outliers, potentially lower coherence
- For 364k documents, try: 50, 100, 200, 300

**n_neighbors:**
- **Higher values** → More global structure preserved, smoother boundaries
- **Lower values** → More local structure, sharper topic boundaries
- Typical range: 10-50

**n_components:**
- **Higher values** → More information preserved from embeddings
- **Lower values** → More aggressive dimensionality reduction
- Typical range: 5-15

## Tips for Large Datasets (100k+ documents)

### For Very Large Datasets like 364k documents:

**Recommended approach - Use sampling first:**

```bash
# Test on a 50k document sample first
.venv/bin/python src/revu/model_evaluator.py \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --embeddings_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --output_dir data/estimand_review/hyperparameter_results \
  --sample_size 50000 \
  --low_memory
```

**Why this works:**
- ✅ Much faster (minutes vs hours per combination)
- ✅ Won't run out of memory
- ✅ Parameters that work on 50k typically work on full dataset
- ✅ Can test many combinations quickly

**Once you find good parameters on the sample, use them on the full dataset:**

```bash
# Run full modeling with best parameters from sample testing
revu model fit \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --output_csv data/final_model.csv \
  --output_dir data/model_output \
  --embeddings_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --min_cluster_size 100 \
  --n_neighbors 30 \
  --n_components 5
```

### General Tips:

1. **Always use pre-computed embeddings** - Computing embeddings once saves significant time

2. **Start with sampling** - For datasets over 100k, use `--sample_size 50000` to test quickly

3. **Use --low_memory flag** - Especially important for large datasets:
   - Slower but more stable
   - Reduces peak memory usage
   - Prevents "killed" errors

4. **Start with a coarse grid** - Test fewer combinations first:
   ```python
   param_grid = ParameterGrid({
       "min_cluster_size": [100, 200],
       "n_neighbors": [30, 50],
       "n_components": [5]
   })
   ```

5. **Refine based on results** - Once you see patterns, test more values in promising ranges

6. **Consider computation time** - Each combination can take 5-30 minutes for large datasets

7. **Run overnight** - For comprehensive searches with many combinations

## Common Issues

### "Process killed" or Memory Error
- Your system ran out of memory during UMAP
- **Solutions:**
  1. Use `--sample_size 50000` to test on a subset
  2. Add `--low_memory` flag
  3. Lower `n_neighbors` values (try 10-30 instead of 50+)
  4. Reduce the parameter grid to fewer combinations
  5. Close other applications to free up RAM
  6. Use a machine with more RAM if available

### Very Low Coherence Scores (<0.3)
- Your texts might be too diverse for topic modeling
- Try preprocessing more aggressively
- Consider if topic modeling is appropriate for your data

### All Combinations Failed
- Check that your CSV file has the correct column name
- Verify embeddings match the document count
- Check logs for specific error messages

## Next Steps

After finding optimal parameters:

1. **Run full modeling with best parameters:**
   ```bash
   revu model fit \
     --input_csv data/your_data.csv \
     --output_csv data/modeled_data.csv \
     --output_dir data/model_output \
     --embeddings_path data/embeddings.npy \
     --model_name "sentence-transformers/all-MiniLM-L6-v2" \
     --min_cluster_size 100 \
     --n_neighbors 30 \
     --n_components 5
   ```

2. **Create visualizations:**
   ```bash
   revu model visualize \
     --input_csv data/modeled_data.csv \
     --model_path data/model_output/bertopic_model \
     --output_dir data/visualizations \
     --embeddings_path data/embeddings.npy
   ```

3. **Analyze topics** - Review the generated visualizations and topic words

4. **Iterate if needed** - If topics don't meet expectations, adjust parameters and rerun
