import click
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer


@click.command()
@click.option('--input_csv', required=True, help='Input CSV with processed_text column')
@click.option('--output_path', required=True, help='Output path for embeddings (.npy file)')
@click.option('--model_name', default="all-MiniLM-L6-v2", help='Embedding model name')
@click.option('--batch_size', default=32, help='Batch size for encoding')
@click.option('--text_column', default='processed_text', help='Column name containing text to embed')
def compute_embeddings(input_csv, output_path, model_name, batch_size, text_column):
    """
    Compute embeddings for documents and save to disk.
    This allows you to compute embeddings once and reuse them for multiple modeling runs.
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")

    texts = df[text_column].astype(str).tolist()
    print(f"✓ Loaded {len(texts)} documents")

    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    print(f"✓ Model loaded")

    print(f"\nComputing embeddings for {len(texts)} documents...")
    print(f"Using batch size: {batch_size}")
    print("(This may take a while for large datasets...)")

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    print(f"\n✓ Embeddings computed with shape: {embeddings.shape}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Memory size: {embeddings.nbytes / (1024**3):.2f} GB")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    print(f"✓ Embeddings saved successfully")

    print("\n" + "="*50)
    print("✅ Embedding computation complete!")
    print("="*50)
    print(f"\nYou can now use these embeddings with:")
    print(f"  revu model fit \\")
    print(f"    --input_csv {input_csv} \\")
    print(f"    --embeddings_path {output_path} \\")
    print(f"    --output_csv your_output.csv")

    return embeddings
