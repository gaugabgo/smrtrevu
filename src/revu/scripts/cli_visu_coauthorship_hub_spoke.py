import click

from revu.scripts.visu_coauthorship_hub_spoke import (
    visualize_global_coauthorship_hub_spoke,
    visualize_topic_coauthorship_hub_spoke,
)


@click.command()
@click.option(
    "--coauthorship-dir", required=True,
    help=(
        "Root coauthorship results directory — must contain "
        "topic_network_summary.csv and topic_{id}/ subdirectories."
    ),
)
@click.option(
    "--top-n-topics", type=int, default=20, show_default=True,
    help="Number of top topics by n_works to include.",
)
@click.option(
    "--top-k-per-topic", type=int, default=10, show_default=True,
    help="Top-K authors per topic by within-topic collaboration strength.",
)
@click.option(
    "--label-top-authors", type=int, default=3, show_default=True,
    help="Number of top authors per community hub to label by name.",
)
@click.option(
    "--topic-info-file", default=None,
    help=(
        "Optional topic_info CSV with 'Topic' and 'Name' columns "
        "for human-readable topic labels."
    ),
)
@click.option(
    "--output", default="coauth_hub_topic.png", show_default=True,
    help="Output PNG path.",
)
@click.option("--figsize-w", type=float, default=22.0, show_default=True, help="Figure width (inches).")
@click.option("--figsize-h", type=float, default=20.0, show_default=True, help="Figure height (inches).")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
def visualize_coauth_topic(
    coauthorship_dir, top_n_topics, top_k_per_topic,
    label_top_authors, topic_info_file, output,
    figsize_w, figsize_h, seed,
):
    """
    Hub-spoke co-authorship network coloured by topic.

    Hubs = per-topic Louvain communities (one hub per topic-community pair).
    Satellites = top-K authors per topic by within-topic strength.
    Colour = topic. Uses per-topic data in {coauthorship_dir}/topic_{id}/.

    Examples:
    ---------
    revu biblio visualize-coauth-topic \\
        --coauthorship-dir data/coauthorship_results \\
        --top-n-topics 20 \\
        --top-k-per-topic 10 \\
        --topic-info-file data/causal_modeled_1Apr2026_topic_info_customlabels.csv \\
        --output data/visualizations/coauth_hub_topic.png
    """
    print("=" * 60)
    print("CO-AUTHORSHIP HUB-SPOKE  (topic-coloured)")
    print("=" * 60)

    visualize_topic_coauthorship_hub_spoke(
        coauthorship_dir=coauthorship_dir,
        top_n_topics=top_n_topics,
        top_k_per_topic=top_k_per_topic,
        label_top_authors=label_top_authors,
        output_file=output,
        topic_info_file=topic_info_file,
        figsize=(figsize_w, figsize_h),
        seed=seed,
    )
    print("\n✅ Done")


@click.command()
@click.option(
    "--coauthorship-dir",
    default="data/coauthorship_results",
    show_default=True,
    help="Root coauthorship results directory containing global network files.",
)
@click.option(
    "--graphml",
    default=None,
    help=(
        "Path to global author_network.graphml. "
        "Defaults to {coauthorship_dir}/author_network.graphml."
    ),
)
@click.option(
    "--metrics",
    default=None,
    help=(
        "Path to global author_network_metrics.csv. "
        "Defaults to {coauthorship_dir}/author_network_metrics.csv."
    ),
)
@click.option(
    "--communities",
    default=None,
    help=(
        "Path to global author_communities.csv. "
        "Defaults to {coauthorship_dir}/author_communities.csv."
    ),
)
@click.option(
    "--top-n", type=int, default=100, show_default=True,
    help="Top-N authors by global strength to include as satellites.",
)
@click.option(
    "--label-top-n", type=int, default=0, show_default=True,
    help=(
        "Number of top authors (by global strength) to label by name. "
        "0 (default) labels every satellite node in the graph."
    ),
)
@click.option(
    "--output", default="coauth_hub_global.png", show_default=True,
    help="Output PNG path.",
)
@click.option("--figsize-w", type=float, default=18.0, show_default=True, help="Figure width (inches).")
@click.option("--figsize-h", type=float, default=16.0, show_default=True, help="Figure height (inches).")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
def visualize_coauth_global(
    coauthorship_dir, graphml, metrics, communities,
    top_n, label_top_n, output, figsize_w, figsize_h, seed,
):
    """
    Hub-spoke co-authorship network coloured by global community.

    Hubs = global Louvain communities (one hub per community with ≥1 top-N author).
    Satellites = top-N authors by global collaboration strength.
    Colour = community. Uses corpus-level data.

    Examples:
    ---------
    revu biblio visualize-coauth-global \\
        --coauthorship-dir data/coauthorship_results \\
        --top-n 100 \\
        --label-top-n 30 \\
        --output data/visualizations/coauth_hub_global.png
    """
    import os
    from pathlib import Path

    base = Path(coauthorship_dir)
    graphml_path = graphml or str(base / "author_network.graphml")
    metrics_path = metrics or str(base / "author_network_metrics.csv")
    communities_path = communities or str(base / "author_communities.csv")

    print("=" * 60)
    print("CO-AUTHORSHIP HUB-SPOKE  (community-coloured)")
    print("=" * 60)

    visualize_global_coauthorship_hub_spoke(
        graphml_file=graphml_path,
        metrics_file=metrics_path,
        communities_file=communities_path,
        top_n=top_n,
        label_top_n=label_top_n if label_top_n > 0 else None,
        output_file=output,
        figsize=(figsize_w, figsize_h),
        seed=seed,
    )
    print("\n✅ Done")
