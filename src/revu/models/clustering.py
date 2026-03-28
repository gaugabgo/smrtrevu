import hdbscan

def get_hdbscan_model(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'):
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True
    )

