from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/abstract_output/processed_EMRnbibparsed.csv")
docs = df['processed_text'].tolist()

# Debug step 1: Check your data
print(f"Number of documents: {len(docs)}")
print(f"Sample document: {docs[0][:200]}...")
print(f"Average document length: {np.mean([len(doc.split()) for doc in docs]):.1f} words")

# Remove any empty/null documents
docs_clean = [doc for doc in docs if doc and len(doc.strip()) > 10]
print(f"Documents after cleaning: {len(docs_clean)}")

# Define potential topics
potential_topics = [
    "Inferential Statistics",
    "Descriptive Statistics", 
    "Bayesian Statistics",
    "Causal Inference",
    "Supervised machine learning",
    "Unsupervised machine learning"
]

# Debug step 2: Try with more relaxed parameters first
print("\n=== Testing with relaxed parameters ===")
topic_model_relaxed = BERTopic(
    embedding_model="thenlper/gte-small",
    min_topic_size=2,  # Reduced from 5
    zeroshot_topic_list=potential_topics,
    zeroshot_min_similarity=0.3,  # Reduced from 0.5
    representation_model=KeyBERTInspired(),
    verbose=True  # Add verbose output
)

topics_relaxed, probabilities = topic_model_relaxed.fit_transform(docs_clean)

print(f"Unique topics found: {len(set(topics_relaxed))}")
print(f"Topic distribution: {pd.Series(topics_relaxed).value_counts()}")

topic_info = topic_model_relaxed.get_topic_info()
print("\nTopic Info:")
print(topic_info)

# Debug step 3: Check topic assignments
print(f"\nTopics assigned: {set(topics_relaxed)}")
print(f"Documents assigned to topics (not -1): {sum(1 for t in topics_relaxed if t != -1)}")

# Debug step 4: If still no results, try without zero-shot first
if len(set(topics_relaxed)) <= 1:  # Only outlier topic (-1)
    print("\n=== Trying standard clustering without zero-shot ===")
    topic_model_standard = BERTopic(
        embedding_model="thenlper/gte-small",
        min_topic_size=2,
        representation_model=KeyBERTInspired()
    )
    topics_standard, _ = topic_model_standard.fit_transform(docs_clean[:100])  # Test with subset
    print(f"Standard clustering found {len(set(topics_standard))} topics")
    print(topic_model_standard.get_topic_info())

# Debug step 5: Try alternative similarity thresholds
print("\n=== Testing different similarity thresholds ===")
for sim_threshold in [0.1, 0.2, 0.3, 0.4]:
    topic_model_test = BERTopic(
        embedding_model="thenlper/gte-small",
        min_topic_size=2,
        zeroshot_topic_list=potential_topics,
        zeroshot_min_similarity=sim_threshold,
        representation_model=KeyBERTInspired()
    )
    topics_test, _ = topic_model_test.fit_transform(docs_clean)
    assigned_count = sum(1 for t in topics_test if t != -1)
    print(f"Similarity {sim_threshold}: {assigned_count}/{len(docs_clean)} documents assigned to topics")

# Debug step 6: Check individual document-topic similarities
print("\n=== Checking document-topic similarities ===")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the same embedding model
model = SentenceTransformer("thenlper/gte-small")

# Embed a few sample documents and topics
sample_docs = docs_clean[:5]
doc_embeddings = model.encode(sample_docs)
topic_embeddings = model.encode(potential_topics)

# Calculate similarities
similarities = cosine_similarity(doc_embeddings, topic_embeddings)

print("Sample document-topic similarities:")
for i, doc in enumerate(sample_docs):
    print(f"\nDoc {i}: {doc[:100]}...")
    for j, topic in enumerate(potential_topics):
        print(f"  {topic}: {similarities[i][j]:.3f}")

# Recommendations based on results
print("\n=== RECOMMENDATIONS ===")
print("1. If no topics found with relaxed parameters:")
print("   - Your similarity threshold might still be too high")
print("   - Your topic labels might not match your document content")
print("   - Consider more specific/detailed topic descriptions")

print("\n2. If some topics found but not many:")
print("   - Gradually increase zeroshot_min_similarity from 0.1")
print("   - Adjust min_topic_size based on your corpus size")
print("   - Try more specific topic names that match your domain")

print("\n3. Alternative topic labels to try:")
alternative_topics = [
    "Statistical analysis and hypothesis testing",
    "Data summarization and descriptive analytics", 
    "Bayesian inference and probabilistic modeling",
    "Causal analysis and intervention effects",
    "Predictive modeling and classification",
    "Clustering and pattern discovery"
]
print("   ", alternative_topics)