def get_top_topics_per_doc(prob_row, top_k=3, threshold=0.05):
    return [i for i, p in enumerate(prob_row) if p >= threshold][:top_k]
