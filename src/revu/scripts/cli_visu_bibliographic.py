import click
import os
from pathlib import Path

# Import the visualizer class
from revu.scripts.visu_bibliographic import BibliographicVisualizer


@click.command()
@click.option('--data_dir', required=True, help='Directory containing CSV files from bibliometric analysis')
@click.option('--output_dir', default=None, help='Directory to save visualizations (defaults to data_dir)')
@click.option('--topic_info_file', default=None, help='CSV file with topic names (must have "Topic" and "Name" columns)')
@click.option('--viz_type',
              type=click.Choice(['all', 'centrality', 'influence', 'trends', 'geo'], case_sensitive=False),
              default='all',
              help='Type of visualization to generate')
# Input file names
@click.option('--centrality_file', default='topic_centrality.csv', help='Filename for centrality data')
@click.option('--influence_file', default='topic_influence.csv', help='Filename for influence data')
@click.option('--trends_file', default='topic_trends.csv', help='Filename for trends data')
# Prevalence filters
@click.option('--top_n_prevalent', type=int, default=None, help='Keep only top N most prevalent topics')
@click.option('--min_works', type=int, default=None, help='Keep only topics with at least this many works')
# Centrality options
@click.option('--top_n_central', type=int, default=None, help='Keep only top N topics by PageRank (recommended: 15-20)')
@click.option('--bottom_n_central', type=int, default=None, help='Keep only bottom N topics by centrality')
@click.option('--centrality_metric', default='mean_pagerank', help='Centrality metric to filter on')
@click.option('--works_file', default=None, help='Path to works CSV with "topic" column for paper satellite nodes in PageRank network')
# Influence filters and options
@click.option('--top_n_influential', type=int, default=None, help='Keep only top N most influential topics')
@click.option('--bottom_n_influential', type=int, default=None, help='Keep only bottom N least influential topics')
@click.option('--influence_metric', default='total_external_citations', help='Influence metric to filter on')
@click.option('--min_citations', type=int, default=None, help='Drop topics with fewer total citations than this floor')
@click.option('--normalize_by_works', is_flag=True, help='Divide citation counts by n_works before plotting (influence graphs)')
# Trend filters and options
@click.option('--top_n_growing', type=int, default=None, help='Keep only top N fastest growing topics')
@click.option('--bottom_n_declining', type=int, default=None, help='Keep only bottom N fastest declining topics')
@click.option('--top_n_changing', type=int, default=None, help='Keep only top N most changing topics (largest absolute slope, either direction)')
@click.option('--top_n_each', type=int, default=None, help='Keep top N fastest growing AND top N fastest declining topics (balanced)')
@click.option('--growing_only', is_flag=True, help='Keep only topics with positive trend')
@click.option('--declining_only', is_flag=True, help='Keep only topics with negative trend')
@click.option('--top_n_spans', type=int, default=25, help='Max rows in temporal spans chart (default: 25)')
@click.option('--sort_spans_by',
              type=click.Choice(['total_works', 'first_year', 'span_length', 'trend_slope'], case_sensitive=False),
              default='total_works',
              help='Sort order for temporal spans chart')
@click.option('--color_growing', default='#2ecc71', help='Hex colour for growing/positive bars (default: #2ecc71)')
@click.option('--color_declining', default='#e74c3c', help='Hex colour for declining/negative bars (default: #e74c3c)')
# Geographic options
@click.option('--metadata_csv', default=None,
              help='Path to metadata CSV containing "authorships.countries" column (pipe-separated alpha-2 codes). '
                   'Required for --viz_type geo. Uses the FULL corpus — no outlier filtering applied.')
@click.option('--geo_cmap', default='YlOrRd',
              help='Colormap for choropleth: any matplotlib name for static, any Plotly scale for interactive (default: YlOrRd)')
@click.option('--geo_format',
              type=click.Choice(['static', 'interactive'], case_sensitive=False),
              default='static',
              help='Output format: "static" saves PNG, "interactive" saves HTML (default: static)')
def visualize_bibliometric(data_dir, output_dir, topic_info_file, viz_type,
                           centrality_file, influence_file, trends_file,
                           top_n_prevalent, min_works,
                           top_n_central, bottom_n_central, centrality_metric, works_file,
                           top_n_influential, bottom_n_influential, influence_metric,
                           min_citations, normalize_by_works,
                           top_n_growing, bottom_n_declining, top_n_changing, top_n_each,
                           growing_only, declining_only,
                           top_n_spans, sort_spans_by, color_growing, color_declining,
                           metadata_csv, geo_cmap, geo_format):
    """
    Generate visualizations from bibliometric analysis results.

    Outlier topics (BERTopic topic=-1 and any custom-labelled "Outlier" topics)
    are automatically filtered before plotting.

    Each chart is saved as an individual file:
      centrality_pagerank_network.png
      influence_citation_activity.png
      influence_internal_vs_external.png
      trends_growth.png
      trends_temporal_spans.png
      geo_publications_map.png / .html

    Supports filtering to focus on the most interesting topics based on:
    - Prevalence (number of works)
    - Centrality (PageRank)
    - Influence (external citations)
    - Trends (growth/decline over time)

    Examples:
    ---------
    # Generate all visualizations with topic names
    revu biblio visualize \\
        --data_dir data/estimand_review/bibliometric_results \\
        --output_dir data/estimand_review/visualizations \\
        --topic_info_file causalinference_modeled_03Jan2026_topic_info.csv

    # PageRank network with paper satellites, top 20 topics
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type centrality \\
        --top_n_central 20 \\
        --works_file data/results/works_with_topics.csv

    # Influence graphs normalised per work, min 50 citations
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type influence \\
        --normalize_by_works \\
        --min_citations 50

    # Trends: top 30 spans sorted by span length, custom colours
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type trends \\
        --top_n_spans 30 \\
        --sort_spans_by span_length \\
        --color_growing '#1d7874' \\
        --color_declining '#c45c45'

    # Geographic map (static PNG)
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type geo \\
        --metadata_csv data/results/works_metadata.csv

    # Geographic map (interactive HTML)
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type geo \\
        --metadata_csv data/results/works_metadata.csv \\
        --geo_format interactive \\
        --geo_cmap Blues
    """
    print("="*60)
    print("BIBLIOMETRIC VISUALIZATION")
    print("="*60)

    # Validate that data_dir is actually a directory, not a file
    if os.path.isfile(data_dir):
        print(f"\n✗ Error: --data_dir must be a directory, not a file.")
        print(f"  You provided: {data_dir}")
        print(f"  Try using: {os.path.dirname(data_dir)}")
        print(f"\nThe CSV filename is specified separately via options like:")
        print(f"  --trends_file (default: topic_trends.csv)")
        print(f"  --centrality_file (default: topic_centrality.csv)")
        print(f"  --influence_file (default: topic_influence.csv)")
        raise click.Abort()

    # Set output directory
    if output_dir is None:
        output_dir = data_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization type: {viz_type}")
    if topic_info_file:
        print(f"Topic info file: {topic_info_file}")
    if not topic_info_file:
        print("  ℹ No --topic_info_file provided; using built-in CUSTOM_LABELS for outlier detection.")

    # Initialize visualizer
    visualizer = BibliographicVisualizer(data_dir=data_dir, topic_info_file=topic_info_file)

    # Track generated files
    generated_files = []

    # ------------------------------------------------------------------ #
    # CENTRALITY                                                           #
    # ------------------------------------------------------------------ #
    try:
        if viz_type in ['all', 'centrality']:
            print("\n" + "="*60)
            print("Generating Centrality Visualization...")
            print("="*60)
            df = visualizer.load_data(centrality_file)

            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if top_n_central or bottom_n_central:
                print("Applying centrality filters...")
                df = visualizer.filter_by_centrality(df, metric=centrality_metric,
                                                     top_n=top_n_central,
                                                     bottom_n=bottom_n_central)

            print(f"Visualizing {len(df)} topics")
            out = os.path.join(output_dir, 'centrality_pagerank_network.png')
            visualizer.visualize_centrality_pagerank_network(df, output_file=out,
                                                              works_file=works_file)
            generated_files.append('centrality_pagerank_network.png')
            print(f"✓ Centrality network saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {centrality_file} not found in {data_dir}, skipping centrality visualization")
    except Exception as e:
        print(f"✗ Error generating centrality visualization: {str(e)}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # INFLUENCE                                                            #
    # ------------------------------------------------------------------ #
    try:
        if viz_type in ['all', 'influence']:
            print("\n" + "="*60)
            print("Generating Influence Visualizations...")
            print("="*60)
            df = visualizer.load_data(influence_file)

            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if min_citations is not None:
                before = len(df)
                total_col = 'total_external_citations'
                if total_col in df.columns:
                    df = df[df[total_col] >= min_citations]
                    print(f"  Applied min_citations filter ({min_citations}): {before - len(df)} topics removed")

            if top_n_influential or bottom_n_influential:
                print("Applying influence filters...")
                df = visualizer.filter_by_influence(df, metric=influence_metric,
                                                    top_n=top_n_influential,
                                                    bottom_n=bottom_n_influential)

            print(f"Visualizing {len(df)} topics")

            out_activity = os.path.join(output_dir, 'influence_citation_activity.png')
            visualizer.visualize_influence_citation_activity(df, output_file=out_activity,
                                                              normalize_by_works=normalize_by_works)
            generated_files.append('influence_citation_activity.png')

            out_scatter = os.path.join(output_dir, 'influence_internal_vs_external.png')
            visualizer.visualize_influence_internal_vs_external(df, output_file=out_scatter,
                                                                 normalize_by_works=normalize_by_works)
            generated_files.append('influence_internal_vs_external.png')
            print(f"✓ Influence visualizations saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {influence_file} not found in {data_dir}, skipping influence visualization")
    except Exception as e:
        print(f"✗ Error generating influence visualization: {str(e)}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # TRENDS                                                               #
    # ------------------------------------------------------------------ #
    try:
        if viz_type in ['all', 'trends']:
            print("\n" + "="*60)
            print("Generating Trends Visualizations...")
            print("="*60)
            df = visualizer.load_data(trends_file)

            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if top_n_growing or bottom_n_declining or top_n_changing or top_n_each or growing_only or declining_only:
                print("Applying trend filters...")
                df = visualizer.filter_by_trends(df, top_n=top_n_growing,
                                                 bottom_n=bottom_n_declining,
                                                 top_n_changing=top_n_changing,
                                                 top_n_each=top_n_each,
                                                 growing_only=growing_only,
                                                 declining_only=declining_only)

            print(f"Visualizing {len(df)} topics")

            out_growth = os.path.join(output_dir, 'trends_growth.png')
            visualizer.visualize_trends_growth(df, output_file=out_growth,
                                               color_growing=color_growing,
                                               color_declining=color_declining)
            generated_files.append('trends_growth.png')

            out_spans = os.path.join(output_dir, 'trends_temporal_spans.png')
            visualizer.visualize_trends_temporal_spans(df, output_file=out_spans,
                                                       top_n=top_n_spans,
                                                       sort_by=sort_spans_by,
                                                       color_growing=color_growing,
                                                       color_declining=color_declining)
            generated_files.append('trends_temporal_spans.png')
            print(f"✓ Trends visualizations saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {trends_file} not found in {data_dir}, skipping trends visualization")
    except Exception as e:
        print(f"✗ Error generating trends visualization: {str(e)}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # GEOGRAPHIC MAP                                                       #
    # ------------------------------------------------------------------ #
    try:
        if viz_type in ['all', 'geo']:
            print("\n" + "="*60)
            print("Generating Geographic Visualization...")
            print("="*60)

            if not metadata_csv:
                print("⚠ Skipping geo map: --metadata_csv is required for geographic visualization.")
            else:
                import pandas as pd
                print(f"  Loading metadata from {metadata_csv}...")
                # Load full corpus — no outlier filtering for geographic counts
                metadata_df = pd.read_csv(metadata_csv, low_memory=False)
                print(f"  Loaded {len(metadata_df)} records")

                ext = 'html' if geo_format == 'interactive' else 'png'
                out_geo = os.path.join(output_dir, f'geo_publications_map.{ext}')
                visualizer.visualize_geo(
                    metadata_df,
                    output_file=out_geo,
                    geo_cmap=geo_cmap,
                    geo_format=geo_format,
                )
                generated_files.append(f'geo_publications_map.{ext}')
                print(f"✓ Geographic map saved")
    except Exception as e:
        print(f"✗ Error generating geographic visualization: {str(e)}")
        import traceback; traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE")
    print("="*60)

    if generated_files:
        print(f"\nGenerated {len(generated_files)} visualization(s):")
        for filename in generated_files:
            print(f"  - {filename}")
        print(f"\nAll files saved to: {output_dir}")
    else:
        print("\n⚠ No visualizations were generated. Check that the required CSV files exist in the data directory.")
