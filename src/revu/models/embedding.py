from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP

def get_embedding_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def get_reduced_embeddings(topic_model, texts):
    """
    Extract and reduce embeddings for visualization from a BERTopic model and a list of texts.
    """
    embeddings = topic_model._extract_embeddings(texts)
    reducer_for_vis = UMAP(n_components=2, metric="cosine", random_state=42)
    reduced_embeddings = reducer_for_vis.fit_transform(embeddings)
    return reduced_embeddings
