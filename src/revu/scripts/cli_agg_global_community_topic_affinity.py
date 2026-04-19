import click

from revu.scripts.agg_global_community_topic_affinity import (
    compute_global_community_topic_affinity,
)


@click.command()
@click.option(
    "--global-communities",
    default="data/coauthorship_results/author_communities.csv",
    show_default=True,
    help="Global author_communities.csv (author_name, community_id).",
)
@click.option(
    "--global-paper-edges",
    default="data/coauthorship_results/author_paper_edges.csv",
    show_default=True,
    help="Global author_paper_edges.csv (id, author_name, ...).",
)
@click.option(
    "--modeled",
    default="data/causal_modeled_25Mar2026.csv",
    show_default=True,
    help="Modeled CSV with 'id' and 'topic' columns.",
)
@click.option(
    "--embeddings",
    default="data/embeddings_2d.npy",
    show_default=True,
    help="2-D UMAP embeddings (.npy); rows must match --modeled.",
)
@click.option(
    "--topic-info",
    default=None,
    help="Topic-info CSV with 'Topic' and 'Name' columns for human-readable labels.",
)
@click.option(
    "--output-dir",
    default="data/coauthorship_results/global_community_topic",
    show_default=True,
    help="Output directory. Writes topic_community_affinity.csv and topic_topic_edges.csv.",
)
@click.option(
    "--top-n-communities",
    type=int,
    default=15,
    show_default=True,
    help="Retain the top-N largest global communities; rest are grouped as 'Other' (-99).",
)
@click.option(
    "--edge-min-weight",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum dot-product weight to keep a cross-topic edge.",
)
def agg_global_community_topic_affinity(
    global_communities,
    global_paper_edges,
    modeled,
    embeddings,
    topic_info,
    output_dir,
    top_n_communities,
    edge_min_weight,
):
    """
    Compute global co-authorship community × topic affinity matrix.

    For each (topic, global_community) pair: how many of the topic's authors
    belong to that community?  Also derives the dominant community per topic
    and pairwise cross-topic edges weighted by shared community overlap.

    Writes two CSVs to --output-dir:
      topic_community_affinity.csv — one row per (topic, community) pair
      topic_topic_edges.csv        — cross-topic edges with weights

    Example
    -------
    revu biblio agg-global-community-topic-affinity \\
      --global-communities data/coauthorship_results/author_communities.csv \\
      --global-paper-edges data/coauthorship_results/author_paper_edges.csv \\
      --modeled            data/causal_modeled_25Mar2026.csv \\
      --embeddings         data/embeddings_2d.npy \\
      --topic-info         data/causal_modeled_25Mar2026_topic_info.csv \\
      --output-dir         data/coauthorship_results/global_community_topic
    """
    import os
    affinity_path = os.path.join(output_dir, "topic_community_affinity.csv")
    edges_path    = os.path.join(output_dir, "topic_topic_edges.csv")

    compute_global_community_topic_affinity(
        global_communities_path=global_communities,
        global_paper_edges_path=global_paper_edges,
        modeled_path=modeled,
        embeddings_path=embeddings,
        output_affinity_path=affinity_path,
        output_edges_path=edges_path,
        topic_info_path=topic_info,
        top_n_communities=top_n_communities,
        edge_min_weight=edge_min_weight,
    )
