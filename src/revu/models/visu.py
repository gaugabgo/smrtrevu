import plotly.express as px
import pandas as pd

def create_document_datamap(reduced_embeddings, topics, texts=None, labels=None):
    """
    Create a 2D document datamap plot (Plotly scatterplot).
    
    Parameters:
    - reduced_embeddings: 2D array or list of shape (n_docs, 2)
    - topics: list or array of topic assignments (len = n_docs)
    - texts: optional list of document texts for tooltips
    - labels: optional dict {topic_id: "Label"} for coloring
    
    Returns:
    - plotly.graph_objects.Figure
    """
    if labels is None or not isinstance(labels, dict):
        labels = {}

    labels_for_plot = [labels.get(int(t), str(t)) for t in topics]

    x, y = zip(*reduced_embeddings)
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "topic": topics,
        "label": labels_for_plot
    })

    if texts is not None:
        df["text"] = texts
    else:
        df["text"] = df["label"]

    fig = px.scatter(
        df, x="x", y="y",
        color="label",
        hover_data={"text": True, "x": False, "y": False, "label": False},
        title="Document Datamap (by Topic)"
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7), selector=dict(mode='markers'))
    fig.update_layout(legend_title_text="Topics")
    return fig
