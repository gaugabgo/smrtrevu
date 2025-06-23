import umap

def get_umap_model(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42):
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
