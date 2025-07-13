import click
import pandas as pd
import os

from revu.models.embedding import get_embedding_model
from revu.models.embedding import get_reduced_embeddings
from revu.models.dimensionality import get_umap_model
from revu.models.clustering import get_hdbscan_model
from revu.models.vectorizer import get_vectorizer_model
from revu.models.representations import get_representation_models
from revu.pipeline.modeling import create_topic_model
from revu.utils.topics import get_top_topics_per_doc
from revu.models.visu import create_document_datamap

@click.group()
def cli():
    pass

@cli.command()
@click.option('--input_csv', required=True)
@click.option('--output_csv', required=True)
@click.option('--output_dir', default="visualizations")
def run_model(input_csv, output_csv, output_dir):
    """
    Run topic modeling with BERTopic
    """
    # Load and preprocess data
    df = pd.read_csv(input_csv)
    texts = df['processed_text'].tolist()

    # Load models
    embedding_model = get_embedding_model()
    embeddings, reduced_embeddings = get_reduced_embeddings(embedding_model, texts)
    umap_model = get_umap_model()
    hdbscan_model = get_hdbscan_model()
    vectorizer_model = get_vectorizer_model()
    representation_model = get_representation_models()

    # Train BERTopic
    topic_model = create_topic_model(
        embedding_model,
        umap_model,
        hdbscan_model,
        vectorizer_model,
        representation_model
    )
    topics, probs = topic_model.fit_transform(texts)

    # Generate labels and assign
    labels = topic_model.generate_topic_labels(nr_words=2, topic_prefix=False, word_length=8, separator=", ")
    topic_model.set_topic_labels(labels)

    # Multi-topic assignments
    multi_topics = [get_top_topics_per_doc(row) for row in probs]
    # Save results
    df["main_topic"] = topics
    df["topic_probs"] = probs.tolist()
    df["multi_topics"] = multi_topics
    print("📋 Checking df before saving...")
    print("DF shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample rows:", df.head(2))

    print("🧪 Checking types:")
    print("type(probs):", type(probs))
    print("type(topics):", type(topics))
    print("Length of probs:", len(probs) if probs is not None else "None")
    print("Length of topics:", len(topics))
    print(f"📁 Writing to {os.path.abspath(output_csv)}")
    df.to_csv(output_csv, index=False)
    print("✅ CSV saved successfully.")
    print(f"Running model on {input_csv}, saving to {output_csv}")
    
    """
    Run visualizations for BERTopic model
    """
    fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
    fig.write_html("BERTopic_barchart.html")

    # Save documents visualization as HTML
    fig_docs = topic_model.visualize_documents(
    texts,
    embeddings=embeddings,
    reduced_embeddings=reduced_embeddings
    )
    fig_docs.write_html("BERTopic_documents.html")
    print("📄 Document visualization saved as BERTopic_documents.html")

    fig_topics = topic_model.visualize_topics(custom_labels=labels)
    fig_topics.write_html("BERTopic_Topics.html")
    print("📄 Topic visualization saved as BERTopic_Topics.html")

    fig_topics_heatmap = topic_model.visualize_heatmap(custom_labels=labels)
    fig_topics_heatmap.write_html("BERTopic_Heatmap.html")
    print("📄 Heatmap visualization saved as BERTopic_Heatmap.html")

    # Plot document datamap
    fig = create_document_datamap(reduced_embeddings, topics, texts=texts, labels=labels)
    fig.write_html("custom_datamap.html")
    fig.write_image("custom_datamap.png", scale=2, width=1000, height=800)

    print(type(topic_model.vectorizer_model))
    print(type(topic_model.umap_model))
    print(type(topic_model.hdbscan_model))
    print(type(topic_model.representation_model))

    # Save model for later evaluation
    model_save_path = os.path.join(output_dir, "bertopic_model")
    os.makedirs(model_save_path, exist_ok=True)

    # Use the embedding model name string if known
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    topic_model.save(
        os.path.join(model_save_path, "bertopic_model"),
        save_embedding_model=embedding_model_name,
        serialization="safetensors"
    )