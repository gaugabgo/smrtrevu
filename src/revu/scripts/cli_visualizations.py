import click
import pandas as pd
import os
import datamapplot
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns


@click.command()
@click.option('--input_csv', required=True, help='Input CSV with topic assignments (output from topic modeling)')
@click.option('--model_path', required=True, help='Path to saved BERTopic model')
@click.option('--output_dir', default="visualizations", help='Directory for visualizations')
@click.option('--embeddings_2d_path', required=True, help='Path to pre-computed 2D embeddings (.npy file). Use "revu model reduce" to create these.')
@click.option('--metadata_csv', default=None, help='Path to separate metadata CSV file with additional fields for hover text (e.g., TI, AU, DP, AB, DOI, citation_count)')
def create_visualizations(input_csv, model_path, output_dir, embeddings_2d_path, metadata_csv):
    """
    Create visualizations from BERTopic model and topic assignments
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)

    # Validate required columns
    required_cols = ['processed_text', 'topic', 'probability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Make sure to run topic modeling first.")

    texts = df['processed_text'].astype(str).tolist()
    topics = df['topic'].tolist()
    print(f"✓ Loaded {len(texts)} documents with topic assignments")

    print(f"Loading BERTopic model from {model_path}...")
    topic_model = BERTopic.load(model_path)
    print("✓ BERTopic model loaded")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Creating visualizations...")
    print("="*50 + "\n")

    # Get document labels from topics
    document_labels = [topic_model.get_topic(topic) for topic in topics]
    document_labels = [
        ", ".join([word for word, _ in label[:3]]) if label else "Outlier"
        for label in document_labels
    ]

    """
    # Generate custom topic labels (use # to remove generic labels code from above)
    custom_labels = {
        -1: "-1",
        0: "Active Transport",
        1: "Air Pollution Exposure",
        2: "Urban Greenspace",
        3: "Childhood Obesity",
        # 
    }

    document_labels = [custom_labels.get(topic, f"Topic {topic}") for topic in topics]
    print(f"Number of mapped labels: {len(document_labels)}")
    print(f"First few mapped labels: {document_labels[:5]}")
    """

    # ========================================
    # LOAD PRE-COMPUTED 2D EMBEDDINGS
    # ========================================
    print(f"Loading pre-computed 2D embeddings from {embeddings_2d_path}...")
    if not os.path.exists(embeddings_2d_path):
        raise FileNotFoundError(
            f"2D embeddings file not found: {embeddings_2d_path}\n"
            f"Please compute 2D embeddings first using:\n"
            f"  revu model reduce --embeddings_path <path> --output_path {embeddings_2d_path}"
        )

    embeddings_2d = np.load(embeddings_2d_path)
    print(f"✓ Loaded 2D embeddings with shape: {embeddings_2d.shape}")

    # Validate embeddings match document count
    if len(embeddings_2d) != len(texts):
        raise ValueError(
            f"2D embeddings count ({len(embeddings_2d)}) doesn't match document count ({len(texts)}). "
            f"Please ensure the embeddings were computed from the same dataset."
        )

    # ========================================
    # BERTOPIC BUILT-IN VISUALIZATIONS
    # ========================================
    print("\n1. Creating BERTopic barchart...")
    fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
    fig.write_html(os.path.join(output_dir, "CausalInference_barchart.html"))
    print("   ✓ Saved BERTopic_barchart.html")

    print("2. Creating BERTopic topics visualization...")
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(os.path.join(output_dir, "CausalInference_BERTopic_Topics.html"))
    print("   ✓ Saved BERTopic_Topics.html")

    print("3. Creating BERTopic heatmap...")
    fig_topics_heatmap = topic_model.visualize_heatmap()
    fig_topics_heatmap.write_html(os.path.join(output_dir, "CausalInference_BERTopic_Heatmap.html"))
    print("   ✓ Saved BERTopic_Heatmap.html")

    # ========================================
    # DATAMAPPLOT STATIC VISUALIZATION
    # ========================================
    
    print("\n4. Creating static datamapplot (outliers filtered)...")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'topic': topics,
        'label': document_labels,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
    })

    # Filter out outliers (topic -1)
    filtered_df = plot_df[plot_df['topic'] != -1]

    filtered_embeddings = filtered_df[['x', 'y']].values
    filtered_labels = filtered_df['label'].tolist()

    fig_docs, ax_docs = datamapplot.create_plot(
        filtered_embeddings,
        filtered_labels,
        title="Topics in the Causal Inference Literature",
        sub_title="Data map of BERTopic-extracted topics",
        label_over_points=False,
        dynamic_label_size=True,
        label_font_size=8,
        label_wrap_width=20,
        figsize=(16, 12)
    )
    static_plot_path = os.path.join(output_dir, "CausalInference_BERTopic_dmp_static.png")
    fig_docs.savefig(static_plot_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved {static_plot_path}")

    # ========================================
    # DATAMAPPLOT INTERACTIVE VISUALIZATION
    # ========================================
    print("\n5. Creating interactive datamapplot...")

    # ========================================
    # LOAD METADATA AND CREATE HOVER TEXT
    # ========================================
    hover_texts = []
    publication_years = None

    if metadata_csv and os.path.exists(metadata_csv):
        print(f"Loading metadata from {metadata_csv}...")
        metadata_df = pd.read_csv(metadata_csv, dtype=str).fillna('')

        # Find ID column in both dataframes (case-insensitive)
        topic_id_col = None
        metadata_id_col = None

        for col in df.columns:
            if col.lower() == 'id':
                topic_id_col = col
                break

        for col in metadata_df.columns:
            if col.lower() == 'id':
                metadata_id_col = col
                break

        if topic_id_col and metadata_id_col:
            # Merge metadata with topic data on ID
            df_with_metadata = df.merge(
                metadata_df,
                left_on=topic_id_col,
                right_on=metadata_id_col,
                how='left',
                suffixes=('', '_meta')
            )
            print(f"✓ Merged metadata for {len(df_with_metadata)} documents")

            # If merge inflated rows (duplicate IDs in metadata), fall back to df directly
            if len(df_with_metadata) != len(df):
                print(f"  Warning: merge produced {len(df_with_metadata)} rows vs {len(df)} documents — using input data directly for hover text")
                df_with_metadata = df.reset_index(drop=True)
            else:
                df_with_metadata = df_with_metadata.reset_index(drop=True)

            # Create rich hover text with metadata fields
            hover_texts = []
            for pos, row in df_with_metadata.iterrows():
                hover_parts = [f"Topic: {document_labels[pos]}"]

                # Add available metadata fields
                if 'title' in row and row['title']:
                    hover_parts.append(f"Title: {row['title']}")
                if 'authorships.raw_author_name' in row and row['authorships.raw_author_name']:
                    hover_parts.append(f"Authors: {row['authorships.raw_author_name']}")
                if 'publication_year' in row and row['publication_year']:
                    hover_parts.append(f"Year: {row['publication_year']}")
                if 'cited_by_count' in row and row['cited_by_count']:
                    hover_parts.append(f"Citation Count: {row['cited_by_count']}")
                if 'DOI' in row and row['DOI']:
                    hover_parts.append(f"Link: https://doi.org/{row['DOI']}")

                hover_texts.append('\n'.join(hover_parts))

    fig_dmp = datamapplot.create_interactive_plot(
        embeddings_2d,
        document_labels,
        hover_text=hover_texts,
        enable_search=True,
        title="Document Topics",
        sub_title="Interactive data map of BERTopic-extracted topics",
        noise_label="Outlier",
        histogram_data=publication_years,
        initial_zoom_fraction=0.9,
    )
    interactive_plot_path = os.path.join(output_dir, "BERTopic_interactive_dmplot.html")
    fig_dmp.save(interactive_plot_path)
    print(f"   ✓ Saved {interactive_plot_path}")

    # ========================================
    # TOPIC PROPORTIONS BY YEAR VISUALIZATION
    # ========================================
    if metadata_csv and os.path.exists(metadata_csv):
        print("\n6. Creating topic proportions by year visualization...")

        # Load metadata if not already loaded
        if 'df_with_metadata' not in locals():
            metadata_df = pd.read_csv(metadata_csv, dtype=str).fillna('')

            # Find ID column in both dataframes (case-insensitive)
            topic_id_col = None
            metadata_id_col = None

            for col in df.columns:
                if col.lower() == 'id':
                    topic_id_col = col
                    break

            for col in metadata_df.columns:
                if col.lower() == 'id':
                    metadata_id_col = col
                    break

            if topic_id_col and metadata_id_col:
                df_with_metadata = df.merge(
                    metadata_df,
                    left_on=topic_id_col,
                    right_on=metadata_id_col,
                    how='left',
                    suffixes=('', '_meta')
                )

        # Check if publication_year column exists
        if 'publication_year' in df_with_metadata.columns:
            # Convert publication_year to numeric, handling any non-numeric values
            df_with_metadata['publication_year'] = pd.to_numeric(
                df_with_metadata['publication_year'],
                errors='coerce'
            )

            # Filter out rows with invalid/missing years
            df_valid_years = df_with_metadata[df_with_metadata['publication_year'].notna()].copy()

            # (1) Total publications by year
            total_pubs_by_year = df_valid_years.groupby('publication_year').size()
            print(f"   → Found publications across {len(total_pubs_by_year)} years")

            # (2) Count topics per year
            topic_counts_by_year = df_valid_years.groupby(['publication_year', 'topic']).size().reset_index(name='count')

            # (3) Calculate proportions (topic publications / total publications * 100)
            topic_proportions = topic_counts_by_year.copy()
            topic_proportions['total_pubs'] = topic_proportions['publication_year'].map(total_pubs_by_year)
            topic_proportions['proportion'] = (topic_proportions['count'] / topic_proportions['total_pubs']) * 100

            # Create the plot
            unique_topics = sorted(df_valid_years['topic'].unique())

            # Filter out outliers (topic -1) from visualization if desired
            topics_to_plot = [t for t in unique_topics if t != -1]

            # Set up color palette
            n_topics = len(topics_to_plot)
            colors = sns.color_palette("husl", n_topics)

            plt.figure(figsize=(14, 8))

            for idx, topic_id in enumerate(topics_to_plot):
                topic_data = topic_proportions[topic_proportions['topic'] == topic_id]

                # Get topic label
                topic_label_list = topic_model.get_topic(topic_id)
                if topic_label_list:
                    topic_label = f"Topic {topic_id}: {', '.join([word for word, _ in topic_label_list[:3]])}"
                else:
                    topic_label = f"Topic {topic_id}"

                plt.plot(
                    topic_data['publication_year'],
                    topic_data['proportion'],
                    marker='o',
                    label=topic_label,
                    color=colors[idx],
                    linewidth=2,
                    markersize=4
                )

            plt.xlabel('Publication Year', fontsize=12)
            plt.ylabel('Topic Proportion (%)', fontsize=12)
            plt.title('Topic Proportions Over Time\n(Topic Publications as % of Total Yearly Publications)', fontsize=14, pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(output_dir, "topic_proportions_by_year.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved {plot_path}")

            # (4) Append yearly proportion values to topic CSV
            # Pivot the data to have years as columns
            proportion_pivot = topic_proportions.pivot(
                index='topic',
                columns='publication_year',
                values='proportion'
            ).fillna(0)

            # Rename columns to indicate they're proportions
            proportion_pivot.columns = [f'proportion_year_{int(year)}' for year in proportion_pivot.columns]
            proportion_pivot = proportion_pivot.reset_index()

            # Load the original topic CSV to append to
            df_output = df.copy()

            # Merge the proportion data with the main dataframe
            df_output = df_output.merge(proportion_pivot, on='topic', how='left')

            # Save updated CSV
            output_csv_path = os.path.join(output_dir, "topics_with_yearly_proportions.csv")
            df_output.to_csv(output_csv_path, index=False)
            print(f"   ✓ Saved updated CSV with yearly proportions: {output_csv_path}")

            # Also create a summary CSV with just topic-level statistics
            topic_summary = topic_proportions.groupby('topic').agg({
                'count': 'sum',
                'proportion': 'mean'
            }).reset_index()
            topic_summary.columns = ['topic', 'total_publications', 'avg_yearly_proportion']

            # Add topic labels
            topic_summary['topic_label'] = topic_summary['topic'].apply(
                lambda t: ', '.join([word for word, _ in topic_model.get_topic(t)[:5]])
                if topic_model.get_topic(t) else 'Outlier'
            )

            # Merge with yearly proportions
            topic_summary = topic_summary.merge(proportion_pivot, on='topic', how='left')

            summary_csv_path = os.path.join(output_dir, "topic_yearly_proportions_summary.csv")
            topic_summary.to_csv(summary_csv_path, index=False)
            print(f"   ✓ Saved topic summary with yearly proportions: {summary_csv_path}")

        else:
            print("   ⚠ Warning: 'publication_year' column not found in metadata. Skipping yearly proportion visualization.")
    else:
        print("\n⚠ Skipping topic proportions by year visualization (no metadata CSV provided)")

    print("\n" + "="*50)
    print("✅ All visualizations created successfully!")
    print("="*50)

    return embeddings_2d
