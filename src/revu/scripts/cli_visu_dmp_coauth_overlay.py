import click

from revu.scripts.visu_dmp_coauth_overlay import create_dmp_with_coauth_overlay


@click.command()
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
    "--community-nodes",
    default="data/within_topic_community_nodes.csv",
    show_default=True,
    help="Community nodes CSV from revu biblio agg-within-topic-community-nodes.",
)
@click.option(
    "--topic-info",
    default=None,
    help="Topic-info CSV with 'Topic' and 'Name' columns for label text.",
)
@click.option(
    "--metadata",
    default=None,
    help="Metadata CSV for hover text (title, authors, year, citations).",
)
@click.option(
    "--output",
    default="data/visualizations/topic_dmp_coauth_overlay.html",
    show_default=True,
    help="Output HTML file path.",
)
@click.option(
    "--title",
    default="Topic Map with Co-authorship Communities",
    show_default=True,
    help="Plot title.",
)
@click.option(
    "--noise-label",
    default="Outlier",
    show_default=True,
    help="Label string used for noise/outlier points.",
)
@click.option(
    "--top-k-communities",
    type=int,
    default=None,
    help="Keep only the top-K communities per topic in the overlay (default: use all from CSV).",
)
def visualize_dmp_coauth_overlay(
    modeled,
    embeddings,
    community_nodes,
    topic_info,
    metadata,
    output,
    title,
    noise_label,
    top_k_communities,
):
    """
    Interactive topic DMP with a toggleable within-topic co-authorship overlay.

    Renders the standard BERTopic data map plot and injects a canvas overlay
    showing co-authorship community bubbles per topic cluster.  Bubbles are
    zoom-synced via DMP's own viewstate change API.  A toggle panel at the
    bottom-left lets users show/hide the overlay.

    Example
    -------
    revu biblio visualize-dmp-coauth-overlay \
      --modeled         data/causal_modeled_25Mar2026.csv \
      --embeddings      data/embeddings_2d.npy \
      --community-nodes data/within_topic_community_nodes.csv \
      --topic-info      data/causal_modeled_1Apr2026_topic_info_customlabels.csv \
      --metadata        data/causal_metadata_validated.csv \
      --output          data/visualizations/topic_dmp_coauth_overlay.html \
      --top-k-communities 3
    """
    create_dmp_with_coauth_overlay(
        modeled_path=modeled,
        embeddings_path=embeddings,
        community_nodes_path=community_nodes,
        output_path=output,
        topic_info_path=topic_info,
        metadata_path=metadata,
        title=title,
        noise_label=noise_label,
        top_k_communities=top_k_communities,
    )
