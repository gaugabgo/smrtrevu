import click
import pandas as pd
import os
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from revu.models.dimensionality import get_umap_model
from revu.models.clustering import get_hdbscan_model
from revu.models.vectorizer import get_vectorizer_model
from revu.models.representations import get_representation_models


@click.command()
@click.option('--input_csv', required=True, help='Input CSV with processed_text column')
@click.option('--output_csv', required=True, help='Output CSV with topic assignments')
@click.option('--output_dir', default="model_output", help='Directory for model and topic info')
@click.option('--model_name', default="all-MiniLM-L6-v2", help='Embedding model name')
@click.option('--embeddings_path', default=None, help='Path to pre-computed embeddings (.npy file). If provided, skips embedding computation.')
@click.option('--save_embeddings', is_flag=True, help='Save computed embeddings to disk for future use')
@click.option('--n_neighbors', default=15, help='UMAP n_neighbors parameter')
@click.option('--n_components', default=5, help='UMAP n_components parameter')
@click.option('--min_cluster_size', default=30, help='HDBSCAN min_cluster_size parameter')
def run_topic_modeling(input_csv, output_csv, output_dir, model_name, embeddings_path, save_embeddings, n_neighbors, n_components, min_cluster_size):
    """
    Run BERTopic topic modeling and save topic assignments to CSV
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)
    texts = df['processed_text'].astype(str).tolist()
    print(f"✓ Loaded {len(texts)} documents")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ========================================
    # HANDLE EMBEDDINGS
    # ========================================
    if embeddings_path and os.path.exists(embeddings_path):
        # Load pre-computed embeddings
        print(f"Loading pre-computed embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings with shape: {embeddings.shape}")

        # Validate embeddings match document count
        if len(embeddings) != len(texts):
            raise ValueError(f"Embeddings count ({len(embeddings)}) doesn't match document count ({len(texts)})")

        embedding_model = model_name  # Use string for model name when using pre-computed embeddings
    else:
        # Compute embeddings
        print(f"Loading embedding model: {model_name}")
        embedding_model = SentenceTransformer(model_name)
        print(f"✓ Model loaded")

        print(f"Computing embeddings for {len(texts)} documents...")
        print("(This may take a while for large datasets...)")
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        print(f"✓ Embeddings computed with shape: {embeddings.shape}")

        # Save embeddings if requested
        if save_embeddings:
            embeddings_save_path = os.path.join(output_dir, "embeddings.npy")
            print(f"Saving embeddings to {embeddings_save_path}...")
            np.save(embeddings_save_path, embeddings)
            print(f"✓ Embeddings saved to {embeddings_save_path}")

    # ========================================
    # CREATE AND FIT BERTOPIC MODEL
    # ========================================
    print("\nCreating BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=get_umap_model(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        ),
        hdbscan_model=get_hdbscan_model(
            min_cluster_size=min_cluster_size,
            metric='euclidean'
        ),
        vectorizer_model=get_vectorizer_model(
            ngram_range=(1, 2),
            stop_words="english"
        ),
        representation_model=get_representation_models(),
        verbose=True
    )
    print("✓ BERTopic model created")

    print(f"\nFitting BERTopic on {len(texts)} documents with pre-computed embeddings...")
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Summary statistics
    num_topics = len(set(topics)) - 1  # -1 to exclude outlier topic
    num_outliers = sum(1 for t in topics if t == -1)
    print(f"\n{'='*50}")
    print(f"✓ Topic modeling complete!")
    print(f"  - Number of topics: {num_topics}")
    print(f"  - Outlier documents: {num_outliers} ({num_outliers/len(topics)*100:.1f}%)")
    print(f"{'='*50}\n")

    # Save results
    df['topic'] = topics
    df['probability'] = probs
    df.to_csv(output_csv, index=False)
    print(f"✓ Results saved to {output_csv}")

    # Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info_path = output_csv.replace('.csv', '_topic_info.csv')
    topic_info.to_csv(topic_info_path, index=False)
    print(f"✓ Topic info saved to {topic_info_path}")

    # ========================================
    # MODEL INFORMATION
    # ========================================
    print("\n" + "="*50)
    print("Model Component Types:")
    print("="*50)
    print(f"Vectorizer: {type(topic_model.vectorizer_model)}")
    print(f"UMAP: {type(topic_model.umap_model)}")
    print(f"HDBSCAN: {type(topic_model.hdbscan_model)}")
    print(f"Representation: {type(topic_model.representation_model)}")

    # ========================================
    # SAVE MODEL
    # ========================================
    print("\n" + "="*50)
    print("Saving BERTopic model...")
    print("="*50)

    model_save_path = os.path.join(output_dir, "bertopic_model")

    # Use the full model name for consistency
    if isinstance(embedding_model, str):
        # Pre-computed embeddings were used
        embedding_model_name = f"sentence-transformers/{model_name}"
    else:
        embedding_model_name = f"sentence-transformers/{model_name}"

    topic_model.save(
        model_save_path,
        save_embedding_model=embedding_model_name,
        serialization="safetensors"
    )
    print(f"✓ Model saved to {model_save_path}")

    print("\n" + "="*50)
    print("✅ Topic modeling complete!")
    print("="*50)

    return topic_model, topics, probs
