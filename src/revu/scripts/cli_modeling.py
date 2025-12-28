import click
import pandas as pd
import os
import datamapplot
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP


from revu.models.embedding import get_embeddings
from revu.models.dimensionality import get_umap_model
from revu.models.clustering import get_hdbscan_model
from revu.models.vectorizer import get_vectorizer_model
from revu.models.representations import get_representation_models
from revu.utils.topics import get_top_topics_per_doc

@click.group()
def cli():
    pass

@cli.command()
@click.option('--input_csv', required=True, help='Input CSV with processed_text column')
@click.option('--output_csv', required=True, help='Output CSV with topic assignments')
@click.option('--output_dir', default="visualizations", help='Directory for visualizations')
@click.option('--model_name', default="all-MiniLM-L6-v2", help='Embedding model name')
@click.option('--n_neighbors', default=15, help='UMAP n_neighbors parameter')
@click.option('--n_components', default=5, help='UMAP n_components parameter')
@click.option('--min_cluster_size', default=30, help='HDBSCAN min_cluster_size parameter')
def run_model(input_csv, output_csv, output_dir, model_name, n_neighbors, n_components, min_cluster_size):
    """
    Run topic modeling with BERTopic
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    texts = df['processed_text'].astype(str).tolist()
    print(f"✓ Loaded {len(texts)} documents")

    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    print(f"✓ Model loaded")

    print("Creating BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,  
        umap_model=get_umap_model(
            n_neighbors=n_neighbors, 
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        ),
        hdbscan_model=get_hdbscan_model(
            min_cluster_size=min_cluster_size,
            metric='euclidean'
        ),
        vectorizer_model=get_vectorizer_model(
            ngram_range=(1, 2),
            stop_words="english"
        ),
        representation_model=get_representation_models(),
        verbose=True
    )
    print("✓ BERTopic model created")

    print(f"Fitting BERTopic on {len(texts)} documents...")
    topics, probs = topic_model.fit_transform(texts)
    
    # Summary statistics
    num_topics = len(set(topics)) - 1  # -1 to exclude outlier topic
    num_outliers = sum(1 for t in topics if t == -1)
    print(f"\n{'='*50}")
    print(f"✓ Topic modeling complete!")
    print(f"  - Number of topics: {num_topics}")
    print(f"  - Outlier documents: {num_outliers} ({num_outliers/len(topics)*100:.1f}%)")
    print(f"{'='*50}\n")
    
    # Save results
    df['topic'] = topics
    df['probability'] = probs
    df.to_csv(output_csv, index=False)
    print(f"✓ Results saved to {output_csv}")

    # Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info_path = output_csv.replace('.csv', '_topic_info.csv')
    topic_info.to_csv(topic_info_path, index=False)
    print(f"✓ Topic info saved to {topic_info_path}")

    """
    Visualizations 
    """
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
            4: "Exercise-Related Heat", 
            5: "Urban Health", 
            6: "Park Recreation", 
            7: "Physical Activity Intervention", 
            8: "Urban Heat", 
            9: "Pedestrian Activity Access", 
            10: "Exercise Intervention for Cancer", 
            11: "Neighborhood Walkability", 
            12: "Diabetes Prevention", 
            13: "Obesity and Physical Activity", 
            14: "Active Travel to School", 
            15: "Maternal Health", 
            16: "Access to Healthy Food", 
            17: "Chronic Pain and Physical Activity", 
            18: "Patient Care", 
            19: "Participatory Research", 
            20: "Intimate Partner Violence", 
            21: "Childhood Physical Activity", 
            22: "Cardiovascular Health", 
            23: "Ecological Health", 
            24: "HIV", 
            25: "Obesity Prevention", 
            26: "Respiratory Disease Rehabilitation", 
            27: "Youth Physical Activity", 
            28: "Lifestyle Cardiovascular Health Intervention", 
            29: "Hypertension Prevalence and Management", 
            30: "Gait", 
            31: "Depression and Anxiety", 
            32: "Noise Pollution", 
            33: "Senior Health Supportive Environment", 
            34: "Physical Activity", 
            35: "Pedestrian Safety", 
            36: "Vaccination", 
            37: "Healthcare Access", 
            38: "Built Environment Intervention and Policy", 
            39: "Neighborhood Health", 
            40: "Dementia and Mild Cognitive Impairment", 
            41: "Batteries", 
            42: "Frailty", 
            43: "Dog Walking" 
            } 


    mapped_labels = [custom_labels.get(topic, f"Topic {topic}") for topic in topics]

    print(f"Number of mapped labels: {len(mapped_labels)}")
    print(f"First few mapped labels: {mapped_labels[:5]}")
    """

   # hover_texts = []
   # for i, row in df.iterrows():
   #      hover_text = f"""
   #      Title: {row['TI']}
   #      Authors: {row['AU']}
   #      Year: {row['DP']}
   #      Abstract: {row['AB'][:200]}{'...' if len(row['AB']) > 200 else ''}
   #      Citation Counts: {row['citation_count']}
   #      Link: https://doi.org/{row['DOI']}"""
   #      hover_texts.append(hover_text)


    # ========================================
    # COMPUTE EMBEDDINGS FOR VISUALIZATIONS
    # ========================================
    print("Computing embeddings for visualizations...")
    
    # Get high-dimensional embeddings
    embeddings_high_dim = embedding_model.encode(
        texts, 
        show_progress_bar=True,
        batch_size=32
    )
    print(f"✓ High-dimensional embeddings shape: {embeddings_high_dim.shape}")
    
    # 2D embeddings
    print("Reducing embeddings to 2D for datamapplot...")
    reducer_2d = UMAP(
        n_components=2, 
        metric='cosine', 
        random_state=42,
        n_neighbors=15,  
        min_dist=0.1     
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings_high_dim)
    print(f"✓ 2D embeddings shape: {embeddings_2d.shape}")

    # ========================================
    # BERTOPIC BUILT-IN VISUALIZATIONS
    # ========================================
    print("\n1. Creating BERTopic barchart...")
    fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
    fig.write_html(os.path.join(output_dir, "BERTopic_barchart_estimand.html"))
    print("   ✓ Saved BERTopic_barchart_estimand.html")

    print("2. Creating BERTopic topics visualization...")
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(os.path.join(output_dir, "BERTopic_Topics_estimand.html"))
    print("   ✓ Saved BERTopic_Topics_estimand.html")

    print("3. Creating BERTopic heatmap...")
    fig_topics_heatmap = topic_model.visualize_heatmap()
    fig_topics_heatmap.write_html(os.path.join(output_dir, "BERTopic_Heatmap_estimand.html"))
    print("   ✓ Saved BERTopic_Heatmap_estimand.html")

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
        title="Estimand Abstracts",
        sub_title="Data map of BERTopic-extracted topics from abstracts related to estimands",
        label_over_points=False,
        dynamic_label_size=True,
        label_font_size=8,  
        label_wrap_width=20,  
        figsize=(16, 12)  
    )
    static_plot_path = os.path.join(output_dir, "BERTopic_documents_estimand.png")
    fig_docs.savefig(static_plot_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved {static_plot_path}")

    # ========================================
    # DATAMAPPLOT INTERACTIVE VISUALIZATION
    # ========================================
    print("\n5. Creating interactive datamapplot...")
    
    # Parse publication years
    publication_years = pd.to_datetime(df['DP'].astype(str), format='%Y', errors='coerce')
    
    # Optional: Create hover text with more info
    hover_texts = [
        f"Topic: {label}\nYear: {year}" 
        for label, year in zip(document_labels, df['DP'].astype(str))
    ]
    
    fig_dmp = datamapplot.create_interactive_plot(
        embeddings_2d,  
        document_labels,  
        hover_text=hover_texts,  
        enable_search=True,
        title="Estimand Abstracts",
        sub_title="Interactive data map of BERTopic-extracted topics from abstracts related to estimands",
        noise_label="Outlier",  
        histogram_data=publication_years,
        # Optional: Uncomment if you have citation counts
        # marker_size_array=df['citation_count'].fillna(0).values,
        # point_radius_min_pixels=2,
        # point_radius_max_pixels=10,
        initial_zoom_fraction=0.9,  
    )
    interactive_plot_path = os.path.join(output_dir, "bertopic_interactive_dmplot_estimand.html")
    fig_dmp.save(interactive_plot_path)
    print(f"   ✓ Saved {interactive_plot_path}")

    # ========================================
    # MODEL INFORMATION
    # ========================================
    print("\n" + "="*50)
    print("Model Component Types:")
    print("="*50)
    print(f"Vectorizer: {type(topic_model.vectorizer_model)}")
    print(f"UMAP: {type(topic_model.umap_model)}")
    print(f"HDBSCAN: {type(topic_model.hdbscan_model)}")
    print(f"Representation: {type(topic_model.representation_model)}")

    # ========================================
    # SAVE MODEL
    # ========================================
    print("\n" + "="*50)
    print("Saving BERTopic model...")
    print("="*50)
    
    model_save_path = os.path.join(output_dir, "bertopic_model_estimand")
    os.makedirs(model_save_path, exist_ok=True)

    # Use the full model name for consistency
    embedding_model_name = f"sentence-transformers/{model_name}"

    topic_model.save(
        os.path.join(model_save_path, "bertopic_model_estimand"),
        save_embedding_model=embedding_model_name,
        serialization="safetensors"
    )
    print(f"✓ Model saved to {model_save_path}")
    
    print("\n" + "="*50)
    print("✅ All visualizations and model saved successfully!")
    print("="*50)
    
    return topic_model, topics, probs, embeddings_2d