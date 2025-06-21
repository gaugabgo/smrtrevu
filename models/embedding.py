from sentence_transformers import SentenceTransformer

def get_embedding_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

# Reduce embeddings for visualization
embeddings = topic_model._extract_embeddings(texts)
reducer_for_vis = UMAP(n_components=2, metric="cosine", random_state=42)
reduced_embeddings = reducer_for_vis.fit_transform(embeddings)