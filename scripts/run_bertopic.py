import pandas as pd
from models.embedding import get_embedding_model
from models.dimensionality import get_umap_model
from models.clustering import get_hdbscan_model
from models.vectorizer import get_vectorizer_model
from models.representations import get_representation_models
from pipeline.topic_modeling import create_topic_model
from utils.topics import get_top_topics_per_doc

# Load and preprocess data
df = pd.read_csv("processed_articles_BE.csv")
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
df.to_csv("BERTopicdf_BE.csv", index=False)

# Save visualizations
fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
fig.write_html("BERTopic_barchart.html")

print("✅ Topic modeling completed and saved.")

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