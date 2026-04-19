import click

from revu.scripts.agg_within_topic_community_nodes import (
    aggregate_within_topic_community_nodes,
)


@click.command()
@click.option(
    "--coauthorship-dir",
    default="data/coauthorship_results",
    show_default=True,
    help="Root directory from revu biblio coauthorship-analysis (contains topic_N/ sub-dirs).",
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
    "--output",
    default="data/within_topic_community_nodes.csv",
    show_default=True,
    help="Output CSV path.",
)
@click.option(
    "--top-k-per-topic",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of community nodes to retain per topic.",
)
@click.option(
    "--min-community-members",
    type=int,
    default=3,
    show_default=True,
    help="Skip communities with fewer than N authors.",
)
@click.option(
    "--top-n-authors",
    type=int,
    default=5,
    show_default=True,
    help="Number of top authors (by strength) to include in the node label.",
)
def agg_within_topic_community_nodes(
    coauthorship_dir,
    modeled,
    embeddings,
    output,
    top_k_per_topic,
    min_community_members,
    top_n_authors,
):
    """
    Aggregate per-topic co-authorship community centroids for DMP overlay.

    Reads community membership and paper–author edges from each topic_N/
    sub-directory, computes mean UMAP position per community, and writes
    a single CSV ready for use as an overlay layer in the topic DMP.

    Example
    -------
    revu biblio agg-within-topic-community-nodes \\
      --coauthorship-dir data/coauthorship_results \\
      --modeled         data/causal_modeled_25Mar2026.csv \\
      --embeddings      data/embeddings_2d.npy \\
      --output          data/within_topic_community_nodes.csv \\
      --top-k-per-topic 3 \\
      --min-community-members 5 \\
      --top-n-authors 5
    """
    aggregate_within_topic_community_nodes(
        coauthorship_dir=coauthorship_dir,
        modeled_path=modeled,
        embeddings_path=embeddings,
        output_path=output,
        top_k_per_topic=top_k_per_topic,
        min_community_members=min_community_members,
        top_n_authors=top_n_authors,
    )
