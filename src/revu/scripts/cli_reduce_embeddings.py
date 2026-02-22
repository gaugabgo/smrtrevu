import click
import pandas as pd
import os
import numpy as np
from umap import UMAP


@click.command()
@click.option('--embeddings_path', required=True, help='Path to high-dimensional embeddings (.npy file)')
@click.option('--output_path', required=True, help='Output path for 2D embeddings (.npy file)')
@click.option('--n_neighbors', default=15, help='UMAP n_neighbors parameter')
@click.option('--min_dist', default=0.1, help='UMAP min_dist parameter')
@click.option('--metric', default='cosine', help='UMAP metric (cosine, euclidean, etc.)')
@click.option('--random_state', default=42, help='Random state for reproducibility')
def reduce_embeddings(embeddings_path, output_path, n_neighbors, min_dist, metric, random_state):
    """
    Reduce high-dimensional embeddings to 2D using UMAP for visualization.
    This allows you to compute 2D embeddings separately to avoid memory issues.
    """
    print("="*70)
    print("REDUCING EMBEDDINGS TO 2D")
    print("="*70)

    print(f"\nLoading high-dimensional embeddings from {embeddings_path}...")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings_high_dim = np.load(embeddings_path)
    print(f"✓ Loaded embeddings with shape: {embeddings_high_dim.shape}")
    print(f"  - Number of documents: {embeddings_high_dim.shape[0]}")
    print(f"  - Embedding dimension: {embeddings_high_dim.shape[1]}")
    print(f"  - Memory size: {embeddings_high_dim.nbytes / (1024**3):.2f} GB")

    print(f"\nReducing embeddings to 2D using UMAP...")
    print(f"Parameters:")
    print(f"  - n_neighbors: {n_neighbors}")
    print(f"  - min_dist: {min_dist}")
    print(f"  - metric: {metric}")
    print(f"  - random_state: {random_state}")
    print("\n(This may take a while for large datasets...)")

    reducer_2d = UMAP(
        n_components=2,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )

    embeddings_2d = reducer_2d.fit_transform(embeddings_high_dim)

    print(f"\n✓ 2D embeddings computed with shape: {embeddings_2d.shape}")
    print(f"  - Memory size: {embeddings_2d.nbytes / (1024**2):.2f} MB")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving 2D embeddings to {output_path}...")
    np.save(output_path, embeddings_2d)
    print(f"✓ 2D embeddings saved successfully")

    print("\n" + "="*70)
    print("✅ 2D embedding reduction complete!")
    print("="*70)
    print(f"\nYou can now use these 2D embeddings with:")
    print(f"  revu model visualize \\")
    print(f"    --embeddings_2d_path {output_path} \\")
    print(f"    --input_csv your_topic_assignments.csv \\")
    print(f"    --model_path your_bertopic_model")

    return embeddings_2d
