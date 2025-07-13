from sentence_transformers import SentenceTransformer
from umap import UMAP

def get_embedding_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def get_reduced_embeddings(embedding_model, texts, show_progress=False):
    """
    Encode and reduce text embeddings for visualization.

    Returns:
        embeddings (np.ndarray): Original high-dimensional embeddings
        reduced_embeddings (np.ndarray): 2D embeddings from UMAP
    """
    embeddings = embedding_model.encode(texts, show_progress_bar=show_progress)
    reducer = UMAP(n_components=2, metric="cosine", random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return embeddings, reduced_embeddings
