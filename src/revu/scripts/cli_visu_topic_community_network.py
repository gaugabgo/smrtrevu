import click

from revu.scripts.visu_topic_community_network import create_topic_community_network


@click.command()
@click.option(
    "--affinity",
    default="data/coauthorship_results/global_community_topic/topic_community_affinity.csv",
    show_default=True,
    help="topic_community_affinity.csv from agg-global-community-topic-affinity.",
)
@click.option(
    "--edges",
    default="data/coauthorship_results/global_community_topic/topic_topic_edges.csv",
    show_default=True,
    help="topic_topic_edges.csv from agg-global-community-topic-affinity.",
)
@click.option(
    "--output",
    default="data/visualizations/topic_community_network.html",
    show_default=True,
    help="Output HTML file path.",
)
@click.option(
    "--title",
    default="Topic Community Network",
    show_default=True,
    help="Figure title.",
)
@click.option(
    "--top-n-edges",
    type=int,
    default=300,
    show_default=True,
    help="Keep only the top-N heaviest edges (0 = keep all).",
)
@click.option(
    "--min-edge-weight",
    type=float,
    default=0.0,
    show_default=True,
    help="Hard minimum edge weight to display.",
)
@click.option(
    "--edge-alpha",
    type=float,
    default=0.35,
    show_default=True,
    help="Edge line opacity (0–1).",
)
def visualize_topic_community_network(
    affinity,
    edges,
    output,
    title,
    top_n_edges,
    min_edge_weight,
    edge_alpha,
):
    """
    Interactive Plotly topic-community network (Layer 3).

    Nodes = topic centroids (UMAP coordinates) coloured by dominant global
    co-authorship community.  Edges = cross-topic connections weighted by
    shared community overlap.  Saved as a standalone HTML file.

    Run agg-global-community-topic-affinity first to produce the input CSVs.

    Example
    -------
    revu biblio visualize-topic-community-network \\
      --affinity data/coauthorship_results/global_community_topic/topic_community_affinity.csv \\
      --edges    data/coauthorship_results/global_community_topic/topic_topic_edges.csv \\
      --output   data/visualizations/topic_community_network.html \\
      --top-n-edges 200
    """
    create_topic_community_network(
        affinity_path=affinity,
        edges_path=edges,
        output_path=output,
        top_n_edges=top_n_edges if top_n_edges > 0 else None,
        min_edge_weight=min_edge_weight,
        edge_alpha=edge_alpha,
        title=title,
    )
