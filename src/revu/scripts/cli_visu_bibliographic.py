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
              type=click.Choice(['all', 'homophily', 'centrality', 'influence', 'trends'], case_sensitive=False),
              default='all',
              help='Type of visualization to generate')
@click.option('--homophily_file', default='topic_homophily.csv', help='Filename for homophily data')
@click.option('--centrality_file', default='topic_centrality.csv', help='Filename for centrality data')
@click.option('--influence_file', default='topic_influence.csv', help='Filename for influence data')
@click.option('--trends_file', default='topic_trends.csv', help='Filename for trends data')
# Prevalence filters
@click.option('--top_n_prevalent', type=int, default=None, help='Keep only top N most prevalent topics')
@click.option('--min_works', type=int, default=None, help='Keep only topics with at least this many works')
# Centrality filters
@click.option('--top_n_central', type=int, default=None, help='Keep only top N topics by centrality (PageRank)')
@click.option('--bottom_n_central', type=int, default=None, help='Keep only bottom N topics by centrality')
@click.option('--centrality_metric', default='mean_pagerank', help='Centrality metric to filter on')
# Homophily filters
@click.option('--top_n_homophilic', type=int, default=None, help='Keep only top N most homophilic topics')
@click.option('--bottom_n_homophilic', type=int, default=None, help='Keep only bottom N least homophilic topics')
@click.option('--min_excess_homophily', type=float, default=None, help='Minimum excess homophily')
# Influence filters
@click.option('--top_n_influential', type=int, default=None, help='Keep only top N most influential topics')
@click.option('--bottom_n_influential', type=int, default=None, help='Keep only bottom N least influential topics')
@click.option('--influence_metric', default='total_external_citations', help='Influence metric to filter on')
# Trend filters
@click.option('--top_n_growing', type=int, default=None, help='Keep only top N fastest growing topics')
@click.option('--bottom_n_declining', type=int, default=None, help='Keep only bottom N fastest declining topics')
@click.option('--growing_only', is_flag=True, help='Keep only topics with positive trend')
@click.option('--declining_only', is_flag=True, help='Keep only topics with negative trend')
def visualize_bibliometric(data_dir, output_dir, topic_info_file, viz_type, homophily_file, centrality_file, influence_file, trends_file,
                          top_n_prevalent, min_works,
                          top_n_central, bottom_n_central, centrality_metric,
                          top_n_homophilic, bottom_n_homophilic, min_excess_homophily,
                          top_n_influential, bottom_n_influential, influence_metric,
                          top_n_growing, bottom_n_declining, growing_only, declining_only):
    """
    Generate visualizations from bibliometric analysis results.

    This command creates publication-quality visualizations of bibliometric
    analysis results including homophily, centrality, influence, and trends.

    Supports filtering to focus on the most interesting topics based on:
    - Prevalence (number of works)
    - Centrality (network importance)
    - Homophily (internal citation patterns)
    - Influence (external citations)
    - Trends (growth/decline over time)

    Examples:
    ---------
    # Generate all visualizations with topic names
    revu biblio visualize \\
        --data_dir data/estimand_review/bibliometric_results \\
        --output_dir data/estimand_review/visualizations \\
        --topic_info_file causalinference_modeled_03Jan2026_topic_info.csv

    # Only show top 15 most prevalent topics
    revu biblio visualize \\
        --data_dir data/results \\
        --topic_info_file topic_info.csv \\
        --top_n_prevalent 15

    # Show only topics with high homophily
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type homophily \\
        --top_n_homophilic 10

    # Show top 10 most central and top 10 most influential topics
    revu biblio visualize \\
        --data_dir data/results \\
        --top_n_central 10 \\
        --top_n_influential 10

    # Show only growing topics with at least 20 works
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type trends \\
        --growing_only \\
        --min_works 20

    # Show fastest growing and declining topics
    revu biblio visualize \\
        --data_dir data/results \\
        --viz_type trends \\
        --top_n_growing 10 \\
        --bottom_n_declining 10
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
        print(f"  --homophily_file (default: topic_homophily.csv)")
        print(f"  --influence_file (default: topic_influence.csv)")
        raise click.Abort()

    # Set output directory
    if output_dir is None:
        output_dir = data_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization type: {viz_type}")
    if topic_info_file:
        print(f"Topic info file: {topic_info_file}")

    # Initialize visualizer
    visualizer = BibliographicVisualizer(data_dir=data_dir, topic_info_file=topic_info_file)

    # Track generated files
    generated_files = []

    try:
        if viz_type in ['all', 'homophily']:
            print("\n" + "="*60)
            print("Generating Homophily Visualization...")
            print("="*60)
            output_file = os.path.join(output_dir, 'homophily_viz.png')
            df = visualizer.load_data(homophily_file)

            # Apply filters
            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if top_n_homophilic or bottom_n_homophilic or min_excess_homophily is not None:
                print("Applying homophily filters...")
                df = visualizer.filter_by_homophily(df, top_n=top_n_homophilic,
                                                    bottom_n=bottom_n_homophilic,
                                                    min_excess=min_excess_homophily)

            print(f"Visualizing {len(df)} topics")
            visualizer.visualize_homophily(df, output_file=output_file)
            generated_files.append('homophily_viz.png')
            print(f"✓ Homophily visualization saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {homophily_file} not found in {data_dir}, skipping homophily visualization")
    except Exception as e:
        print(f"✗ Error generating homophily visualization: {str(e)}")

    try:
        if viz_type in ['all', 'centrality']:
            print("\n" + "="*60)
            print("Generating Centrality Visualization...")
            print("="*60)
            output_file = os.path.join(output_dir, 'centrality_viz.png')
            df = visualizer.load_data(centrality_file)

            # Apply filters
            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if top_n_central or bottom_n_central:
                print("Applying centrality filters...")
                df = visualizer.filter_by_centrality(df, metric=centrality_metric,
                                                     top_n=top_n_central,
                                                     bottom_n=bottom_n_central)

            print(f"Visualizing {len(df)} topics")
            visualizer.visualize_centrality(df, output_file=output_file)
            generated_files.append('centrality_viz.png')
            print(f"✓ Centrality visualization saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {centrality_file} not found in {data_dir}, skipping centrality visualization")
    except Exception as e:
        print(f"✗ Error generating centrality visualization: {str(e)}")

    try:
        if viz_type in ['all', 'influence']:
            print("\n" + "="*60)
            print("Generating Influence Visualization...")
            print("="*60)
            output_file = os.path.join(output_dir, 'influence_viz.png')
            df = visualizer.load_data(influence_file)

            # Apply filters
            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if top_n_influential or bottom_n_influential:
                print("Applying influence filters...")
                df = visualizer.filter_by_influence(df, metric=influence_metric,
                                                    top_n=top_n_influential,
                                                    bottom_n=bottom_n_influential)

            print(f"Visualizing {len(df)} topics")
            visualizer.visualize_influence(df, output_file=output_file)
            generated_files.append('influence_viz.png')
            print(f"✓ Influence visualization saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {influence_file} not found in {data_dir}, skipping influence visualization")
    except Exception as e:
        print(f"✗ Error generating influence visualization: {str(e)}")

    try:
        if viz_type in ['all', 'trends']:
            print("\n" + "="*60)
            print("Generating Trends Visualization...")
            print("="*60)
            output_file = os.path.join(output_dir, 'trends_viz.png')
            df = visualizer.load_data(trends_file)

            # Apply filters
            if top_n_prevalent or min_works:
                print("Applying prevalence filters...")
                df = visualizer.filter_by_prevalence(df, top_n=top_n_prevalent, min_works=min_works)

            if top_n_growing or bottom_n_declining or growing_only or declining_only:
                print("Applying trend filters...")
                df = visualizer.filter_by_trends(df, top_n=top_n_growing,
                                                bottom_n=bottom_n_declining,
                                                growing_only=growing_only,
                                                declining_only=declining_only)

            print(f"Visualizing {len(df)} topics")
            visualizer.visualize_trends(df, output_file=output_file)
            generated_files.append('trends_viz.png')
            print(f"✓ Trends visualization saved")
    except FileNotFoundError:
        print(f"⚠ Warning: {trends_file} not found in {data_dir}, skipping trends visualization")
    except Exception as e:
        print(f"✗ Error generating trends visualization: {str(e)}")

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
