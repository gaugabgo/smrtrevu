# Quick Start Guide for 364k Document Dataset

This guide provides the exact commands you need to run for your 364,344 document dataset.

## Your Data

- **Input CSV**: `data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv`
- **Embeddings**: `data/estimand_review/embeddings.npy` (needs to be computed first)
- **Total documents**: 364,344

## Step-by-Step Workflow

### Step 1: Compute Embeddings (One-time, ~50 minutes)

```bash
revu model embed \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --output_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --batch_size 32
```

**Expected output:**
- `embeddings.npy` file (~550 MB)
- Should take 45-60 minutes on CPU

### Step 2: Find Optimal Hyperparameters (Test on Sample)

**Important:** Don't run on full dataset! Use a 50k sample to find good parameters quickly.

```bash
.venv/bin/python src/revu/model_evaluator.py \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --embeddings_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --output_dir data/estimand_review/hyperparameter_results \
  --sample_size 20000 \
  --low_memory
```

**What this does:**
- Randomly samples 50,000 documents
- Tests 18 hyperparameter combinations
- Takes ~2-4 hours total
- Won't run out of memory

**Output files:**
- `hyperparameter_results_TIMESTAMP.csv` - All results
- `best_params_TIMESTAMP.txt` - Best parameters

### Step 3: Run Full Topic Modeling with Best Parameters

Once you have the best parameters from Step 2 (let's say they were: `min_cluster_size=100, n_neighbors=30, n_components=5`), run on the full dataset:

```bash
revu model fit \
  --input_csv data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv \
  --output_csv data/estimand_review/causalinference_modeled_02Jan2026.csv \
  --output_dir data/estimand_review \
  --embeddings_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --n_neighbors 30 \
  --n_components 5 \
  --min_cluster_size 100
```

**Expected behavior:**
- Loads pre-computed embeddings (fast)
- Runs UMAP (may take 30-60 minutes, could still crash if not enough RAM)
- Runs HDBSCAN clustering
- Saves topic assignments and model

**If this still crashes:**
You may need to run on a machine with more RAM (16GB+ recommended for 364k documents) or consider further sampling.

### Step 4: Create Visualizations

```bash
revu model visualize \
  --input_csv data/estimand_review/causalinference_modeled_02Jan2026.csv \
  --model_path data/estimand_review/bertopic_model \
  --output_dir data/estimand_review/visualizations \
  --embeddings_path data/estimand_review/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2"
```

## Memory Requirements

For your 364k dataset:
- **Embeddings**: ~550 MB
- **UMAP processing**: ~4-6 GB RAM peak
- **Total system RAM needed**: 8-12 GB minimum, 16GB+ recommended

## If You Still Get "Process Killed" Errors

### Option A: Use Smaller Sample for Full Workflow

If even the full dataset fails, run everything on a 100k sample:

```bash
# 1. Create a sample CSV first
head -n 100001 data/estimand_review/processed_abstract_deduplicated_30Dec2025.csv > data/estimand_review/sample_100k.csv

# 2. Compute embeddings for sample
revu model embed \
  --input_csv data/estimand_review/sample_100k.csv \
  --output_path data/estimand_review/embeddings_100k.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2"

# 3. Test hyperparameters on sample
.venv/bin/python src/revu/model_evaluator.py \
  --input_csv data/estimand_review/sample_100k.csv \
  --embeddings_path data/estimand_review/embeddings_100k.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --output_dir data/estimand_review/hyperparameter_results \
  --low_memory

# 4. Run full modeling on sample with best params
revu model fit \
  --input_csv data/estimand_review/sample_100k.csv \
  --output_csv data/estimand_review/modeled_100k.csv \
  --output_dir data/estimand_review \
  --embeddings_path data/estimand_review/embeddings_100k.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --n_neighbors 30 \
  --n_components 5 \
  --min_cluster_size 50
```

### Option B: Use a Cloud Instance

For full 364k dataset, consider using:
- AWS EC2 (e.g., r5.xlarge with 32GB RAM)
- Google Colab Pro (with high-RAM runtime)
- Your university's computing cluster

## Customizing Hyperparameter Search

Edit `src/revu/model_evaluator.py` line 86-90 to test different parameters:

```python
param_grid = ParameterGrid({
    "min_cluster_size": [50, 75, 100, 150],     # Customize these
    "n_neighbors": [20, 30, 40],                # Customize these
    "n_components": [5, 7, 10]                  # Customize these
})
```

For 364k documents, recommended starting ranges:
- `min_cluster_size`: [100, 150, 200, 250]
- `n_neighbors`: [20, 30, 50]
- `n_components`: [5, 10]

## Expected Timeline

| Step | Time (364k docs) | Time (50k sample) |
|------|-----------------|-------------------|
| Compute embeddings | 45-60 min | 10-15 min |
| Hyperparameter search | N/A (use sample!) | 2-4 hours |
| Full topic modeling | 1-2 hours* | 15-30 min |
| Visualizations | 30-60 min | 10-15 min |

*May crash if insufficient RAM

## Troubleshooting

**"Process killed" during embedding:**
- Reduce `--batch_size` to 16 or 8
- Close other applications

**"Process killed" during UMAP (hyperparameter search):**
- ✅ Use `--sample_size 50000`
- ✅ Add `--low_memory` flag
- This should fix it!

**"Process killed" during full topic modeling:**
- Parameters that work on 50k sample usually work on full dataset
- If still failing, you need more RAM
- Consider using 100k or 200k sample instead of full dataset

**Results don't look good:**
- Try different hyperparameters
- Check if your text preprocessing is appropriate
- Consider if topic modeling is right for your data
