from bertopic import BERTopic

def create_topic_model(embedding_model, umap_model, hdbscan_model, vectorizer_model, representation_model):
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True
    )
