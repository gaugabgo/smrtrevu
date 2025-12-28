from transformers.pipelines import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer

def get_embeddings(texts, model_name="allenai/scibert_scivocab_uncased", batch_size=32):
    """
    Extract embeddings using sentence-transformers.
    No dimensionality reduction - BERTopic will handle that with UMAP.
    
    Args:
        texts: List of text documents
        model_name: HuggingFace model identifier
        batch_size: Batch size for encoding (adjust based on GPU memory)
        
    Returns:
        embeddings (np.ndarray): High-dimensional embeddings
    """
    print(f"Loading model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    print(f"Extracting embeddings for {len(texts)} documents...")
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

