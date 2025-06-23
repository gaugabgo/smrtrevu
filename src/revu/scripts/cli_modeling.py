import click
import pandas as pd


from revu.models.embedding import get_embedding_model
from revu.models.dimensionality import get_umap_model
from revu.models.clustering import get_hdbscan_model
from revu.models.vectorizer import get_vectorizer_model
from revu.models.representations import get_representation_models
from revu.pipeline.modeling import create_topic_model
from revu.utils.topics import get_top_topics_per_doc

@click.group()
def cli():
    pass

@cli.command()
@click.option('--input_csv', required=True)
@click.option('--output_csv', required=True)
def run_model(input_csv, output_csv):
    """
    Run topic modeling with BERTopic
    """
    # Load and preprocess data
    df = pd.read_csv(input_csv)
    texts = df['processed_text'].tolist()

    # Load models
    embedding_model = get_embedding_model()
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
    df.to_csv(output_csv, index=False)
    print(f"Running model on {input_csv}, saving to {output_csv}")

@cli.command()
@click.option('--input_csv', required=True)
@click.option('--output_dir', default="visualizations")
def visualize(input_csv, output_dir):
    """
    Run visualizations for BERTopic model
    """

    fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
    fig.write_html("BERTopic_barchart.html")

    # Save documents visualization as HTML
    fig_docs = topic_model.visualize_documents(texts, embeddings=reduced_embeddings, custom_labels=topic_labels)
    fig_docs.write_html("BERTopic_documents.html")
    print("📄 Document visualization saved as BERTopic_documents.html")

    fig_topics = topic_model.visualize_topics(custom_labels=topic_labels)
    fig_topics.write_html("BERTopic_Topics.html")
    print("📄 Topic visualization saved as BERTopic_Topics.html")

    fig_topics_heatmap = topic_model.visualize_heatmap(custom_labels=topic_labels)
    fig_topics_heatmap.write_html("BERTopic_Heatmap.html")
    print("📄 Heatmap visualization saved as BERTopic_Heatmap.html")

    fig_datamap = topic_model.visualize_document_datamap(texts, embeddings=reduced_embeddings, custom_labels=topic_labels)
    fig_datamap.savefig("BERTopic_Datamap.png", dpi=300)
    print("📄 Document visualization saved as BERTopic_Datamap.png")
    print(f"Generating visualizations from {input_csv}, saving to {output_dir}")
