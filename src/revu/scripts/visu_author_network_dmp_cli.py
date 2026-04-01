import click

from revu.scripts.visu_author_network_dmp import visualize_author_dmp


@click.command()
@click.option('--embeddings', default='data/embeddings_2d.npy', show_default=True,
              help='Path to embeddings_2d.npy (2-D UMAP coordinates)')
@click.option('--modeled', default='data/causal_modeled_25Mar2026.csv', show_default=True,
              help='Path to modeled CSV with id and topic columns')
@click.option('--metadata', default='data/metadata_deduplicated.csv', show_default=True,
              help='Path to metadata CSV with id and authorships.raw_author_name columns')
@click.option('--communities', default='data/coauthorship_results/author_communities.csv', show_default=True,
              help='Path to author_communities.csv')
@click.option('--metrics', default='data/coauthorship_results/author_network_metrics.csv', show_default=True,
              help='Path to author_network_metrics.csv')
@click.option('--topic-info', default=None,
              help='Optional topic_info CSV for named community labels')
@click.option('--min-papers', type=int, default=5, show_default=True,
              help='Exclude authors with fewer than N papers')
@click.option('--top-label-n', type=int, default=200, show_default=True,
              help='Number of top authors to label by community')
@click.option('--output', default='data/visualizations/author_topic_communities_dmp.html', show_default=True,
              help='Output file (.html for interactive, .png/.pdf for static)')
@click.option('--static', is_flag=True, default=False,
              help='Produce a static image instead of the default interactive HTML')
@click.option('--title', default='Author Co-authorship Map', show_default=True,
              help='Plot title')
def visualize_author_network_dmp(embeddings, modeled, metadata, communities, metrics,
                                  topic_info, min_papers, top_label_n, output, static, title):
    """Author co-authorship map via DataMapPlot (UMAP-positioned, interactive by default)."""
    visualize_author_dmp(
        embeddings_path=embeddings,
        modeled_path=modeled,
        metadata_path=metadata,
        communities_path=communities,
        metrics_path=metrics,
        topic_info_path=topic_info,
        min_papers=min_papers,
        top_label_n=top_label_n,
        output_file=output,
        title=title,
        interactive=not static,
    )
