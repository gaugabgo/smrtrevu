import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from umap import UMAP
import plotly.io as pio

# Use plotly's 'browser' mode for VSCode or export
pio.renderers.default = 'browser'

# Load data
df_corpus = pd.read_csv('processed_articles_BE.csv')
texts = df_corpus['processed_text'].tolist()

# Define models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

umap_model = umap.UMAP(n_neighbors=15, 
                       n_components=5, 
                       min_dist=0.0, 
                       metric='cosine',
                       random_state=42)

hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=100, 
                                metric='euclidean', 
                                cluster_selection_method='eom', 
                                prediction_data=True)

representation_models = [
    MaximalMarginalRelevance(diversity=0.3),
    KeyBERTInspired(),
]

vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

# Create BERTopic model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_models,
    calculate_probabilities=True,
    verbose=True
)

# Fit the model
topics, probs = topic_model.fit_transform(texts)

# Fit the model
topics, probs = topic_model.fit_transform(texts)

# Generate list of new labels
topic_labels_list = topic_model.generate_topic_labels(nr_words=2, 
                                                      topic_prefix=False,
                                                      word_length=8,
                                                      separator=", ")

# Set the topic labels directly
topic_model.set_topic_labels(topic_labels_list)

# Multi-topic assignment
def get_top_topics_per_doc(prob_row, top_k=3, threshold=0.05):
    return [i for i, p in enumerate(prob_row) if p >= threshold][:top_k]

multi_topic_assignments = [get_top_topics_per_doc(prob_row) for prob_row in probs]

# Save to DataFrame
df_corpus['main_topic'] = topics
df_corpus['topic_probs'] = probs.tolist()
df_corpus['multi_topics'] = multi_topic_assignments

# Save processed CSV
output_file = '/Users/gabriellegauthier/Projects/revu/BERTopicdf_BE.csv'
df_corpus.to_csv(output_file, index=False)
print(f"✅ Processed CSV saved to {output_file}")

# Save barchart visualization as HTML
fig_bar = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
fig_bar.write_html("BERTopic_barchart.html")
print("📊 Barchart saved as BERTopic_barchart.html")

# Reduce embeddings for visualization
embeddings = topic_model._extract_embeddings(texts)
reducer_for_vis = UMAP(n_components=2, metric="cosine", random_state=42)
reduced_embeddings = reducer_for_vis.fit_transform(embeddings)

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