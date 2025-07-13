from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.model_selection import ParameterGrid
import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def main():
    # Load and preprocess texts
    df = pd.read_csv("data/preprocess_out/processed_articles_BE.csv")
    texts = df['processed_text'].tolist()

    # Load sentence-transformers model and compute embeddings once
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # defensive programming
    if embeddings is None:
        raise Exception("crash" \
        "")

    # Define static models
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

    # Representations
    keybert_model = KeyBERTInspired()
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {
        "KeyBERT": keybert_model,
        "MMR": mmr_model
    }

    # Gensim prep
    # Tokenize the column (assumes whitespace-separated tokens)
    texts_tokenized = df['processed_text'].astype(str).apply(str.split).tolist()

    # Prepare for BERTopic — still need untokenized strings here
    texts_for_bertopic = [" ".join(doc) for doc in texts_tokenized]

    #Gensim dictionary
    dictionary = Dictionary(texts_tokenized)
    corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

    # Parameter grid
    param_grid = ParameterGrid({
        "min_cluster_size": [15, 30, 50, 100, 200],
        "n_neighbors": [5, 10, 15]
    })

    results = []

    # Iterate through parameter grid
    for params in param_grid:
        min_cluster = params["min_cluster_size"]
        neighbors = params["n_neighbors"]
        
        # Update UMAP and HDBSCAN for each parameter set
        umap_model = UMAP(n_neighbors=neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        
        # Initialize topic model
        topic_model = BERTopic(
            embedding_model=embedding_model,  # No need to re-embed; we're passing embeddings
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            verbose=False
        )
        
        topics, _ = topic_model.fit_transform(texts_for_bertopic, embeddings)

        # Get topic words
        topic_words = topic_model.get_topics()
        topic_words = [[word for word, _ in topic_words[i]] for i in topic_words]
        
        print(f"Params: {params}, #Topics: {len(topic_words)}")

        # Compute coherence
        cm = CoherenceModel(
            topics=topic_words,
            texts=texts_tokenized,
            dictionary=dictionary,
            coherence='c_v'  # Other options: 'u_mass', 'c_uci', 'c_npmi'
        )
        coherence = cm.get_coherence()

        results.append({
            "min_cluster_size": min_cluster,
            "n_neighbors": neighbors,
            "coherence": coherence
        })

    # Output results
    df_results = pd.DataFrame(results)
    print(df_results.sort_values(by="coherence", ascending=False))

if __name__ == '__main__':
    main()