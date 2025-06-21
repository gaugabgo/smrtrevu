import hdbscan

def get_hdbscan_model(min_cluster_size=100, metric='euclidean'):
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method='eom',
        prediction_data=True
    )

