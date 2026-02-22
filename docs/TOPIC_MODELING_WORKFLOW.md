# Topic Modeling Workflow Guide

This guide explains how to use the split topic modeling commands for memory-efficient processing of large datasets.

## Overview

The topic modeling functionality has been split into three separate commands:

1. **`revu model embed`** - Compute and save embeddings
2. **`revu model fit`** - Run BERTopic topic modeling
3. **`revu model visualize`** - Create visualizations

## Why Split the Process?

For large datasets (100k+ documents), running everything in one go can cause memory issues. By splitting the process:

- **Embeddings are computed once** and saved to disk
- **UMAP/HDBSCAN can be rerun** with different parameters without recomputing embeddings
- **Visualizations can be regenerated** without rerunning the model
- **Process can resume** if it crashes during UMAP/HDBSCAN

## Recommended Workflow for Large Datasets

### Step 1: Compute Embeddings

```bash
revu model embed \
  --input_csv data/processed_data.csv \
  --output_path data/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --batch_size 32
```

**What this does:**
- Loads your processed text data
- Computes embeddings using the specified model
- Saves embeddings to disk as a NumPy array (.npy file)
- This is the most memory-intensive step but runs stably

**Options:**
- `--batch_size`: Adjust based on your GPU memory (default: 32)
- `--text_column`: Specify if your text column isn't named "processed_text"

### Step 2: Run Topic Modeling

```bash
revu model fit \
  --input_csv data/processed_data.csv \
  --output_csv data/modeled_data.csv \
  --output_dir data/model_output \
  --embeddings_path data/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --n_neighbors 50 \
  --min_cluster_size 100
```

**What this does:**
- Loads pre-computed embeddings from Step 1
- Runs UMAP dimensionality reduction
- Runs HDBSCAN clustering
- Assigns topics to documents
- Saves results to CSV with `topic` and `probability` columns
- Saves the trained BERTopic model

**Key Parameters:**
- `--n_neighbors`: Controls UMAP locality (higher = more global structure)
- `--min_cluster_size`: Minimum documents per topic (higher = fewer, larger topics)
- `--n_components`: UMAP output dimensions (default: 5)

**If this crashes:**
You can adjust UMAP parameters and rerun without recomputing embeddings!

### Step 3: Create Visualizations

```bash
revu model visualize \
  --input_csv data/modeled_data.csv \
  --model_path data/model_output/bertopic_model \
  --output_dir data/visualizations \
  --embeddings_path data/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2"
```

**What this does:**
- Loads the trained model and topic assignments
- Loads or computes embeddings
- Creates 2D UMAP projection for visualization
- Generates BERTopic built-in visualizations (barchart, heatmap, etc.)
- Creates interactive and static datamapplot visualizations

**Outputs:**
- `BERTopic_barchart.html` - Top words per topic
- `BERTopic_Topics.html` - Topic similarity map
- `BERTopic_Heatmap.html` - Topic similarity heatmap
- `BERTopic_documents_static.png` - Static 2D document map
- `BERTopic_interactive_dmplot.html` - Interactive document explorer

## Alternative: All-in-One Workflow

For smaller datasets or when you want everything at once:

```bash
revu model run \
  --input_csv data/processed_data.csv \
  --output_csv data/modeled_data.csv \
  --output_dir data/visualizations \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --n_neighbors 15 \
  --min_cluster_size 30
```

This runs all steps in sequence but doesn't save intermediate embeddings.

## Tips for Large Datasets

1. **Monitor memory usage** - Use `htop` or Activity Monitor during embedding computation
2. **Use appropriate batch size** - Lower if you run out of GPU memory
3. **Start with sampling** - Test parameters on a smaller random sample first
4. **Increase min_cluster_size** - For very large datasets, use 100+ to get manageable topic counts
5. **Save embeddings** - Always use `--save_embeddings` or `revu model embed` for datasets over 50k documents

## Experimenting with Parameters

Once embeddings are computed, you can quickly test different UMAP/HDBSCAN parameters:

```bash
# Try different clustering parameters
for min_size in 50 100 200; do
  revu model fit \
    --input_csv data/processed_data.csv \
    --output_csv data/modeled_data_${min_size}.csv \
    --output_dir data/model_output_${min_size} \
    --embeddings_path data/embeddings.npy \
    --model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --min_cluster_size ${min_size}
done
```

## Memory Estimates

For reference, with `all-MiniLM-L6-v2` (384 dimensions):
- **Embeddings**: ~1.5 GB per 1M documents
- **UMAP (5D)**: Additional ~200 MB per 1M documents
- **Peak during HDBSCAN**: 3-4x the UMAP output size

For 364k documents:
- Embeddings: ~550 MB
- UMAP working memory: ~2-3 GB
- Total peak: ~4-5 GB

## Troubleshooting

**"Process killed" during UMAP:**
- Your system ran out of memory
- Try increasing swap space or using a machine with more RAM
- Consider using a smaller `n_neighbors` value (e.g., 30 instead of 50)

**Embeddings don't match document count:**
- Ensure you're using the same input CSV for all steps
- Don't modify the CSV between steps

**Model won't load:**
- Ensure you're using the same embedding model name in all steps
- Check that the model path is correct
