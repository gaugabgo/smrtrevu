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
import logging
import numpy as np

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Load and preprocess texts
    df = pd.read_csv("data/estimand_review/processed_estimand_urbanhealthpolicy_24Oct2025.csv")
    texts_for_bertopic = df['processed_text'].astype(str).tolist()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    #SentenceTransformer("allenai/scibert_scivocab_uncased")

    print("Computing embeddings...")
    embeddings = embedding_model.encode(
        texts_for_bertopic, 
        show_progress_bar=True,
        batch_size=32 
    )
    print(f"Embeddings shape: {embeddings.shape}")

    vectorizer_model = CountVectorizer(stop_words="english", min_df=1, max_df=1.0, ngram_range=(1, 2))

    keybert_model = KeyBERTInspired()
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {
        "KeyBERT": keybert_model,
        "MMR": mmr_model
    }

    texts_tokenized = [text.split() for text in texts_for_bertopic]

    dictionary = Dictionary(texts_tokenized)
    corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

    param_grid = ParameterGrid({
        "min_cluster_size": [15, 30, 50, 100, 200],
        "n_neighbors": [5, 10, 15]
    })

    results = []
    for params in param_grid:

        min_cluster = params["min_cluster_size"]
        neighbors = params["n_neighbors"]

        umap_model = UMAP(n_neighbors=neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            verbose=False
        )

        try:
            topics, _ = topic_model.fit_transform(texts_for_bertopic, embeddings)

            topic_words = topic_model.get_topics()
            topic_words = [[word for word, _ in topic_words[i]] for i in topic_words]
            
            print(f"✓ Params: {params}, #Topics: {len(topic_words)}")

            cm = CoherenceModel(
                topics=topic_words,
                texts=texts_tokenized,
                dictionary=dictionary,
                coherence='c_v'  
            )
            coherence = cm.get_coherence()

            results.append({
                "min_cluster_size": min_cluster,
                "n_neighbors": neighbors,
                "coherence": coherence,
                "num_topics": len(topic_words),
                "status": "success"
            })
            
        except ValueError as e:
            if "max_df corresponds to < documents than min_df" in str(e):
                logger.warning(f"✗ Params {params}: Clusters too small for vectorization. Skipping.")
                results.append({
                    "min_cluster_size": min_cluster,
                    "n_neighbors": neighbors,
                    "coherence": None,
                    "num_topics": None,
                    "status": "failed_small_clusters"
                })
            else:
                logger.error(f"✗ Params {params}: Unexpected ValueError - {e}")
                results.append({
                    "min_cluster_size": min_cluster,
                    "n_neighbors": neighbors,
                    "coherence": None,
                    "num_topics": None,
                    "status": f"failed_other: {str(e)}"
                })
        except Exception as e:
            logger.error(f"✗ Params {params}: Unexpected error - {type(e).__name__}: {e}")
            results.append({
                "min_cluster_size": min_cluster,
                "n_neighbors": neighbors,
                "coherence": None,
                "num_topics": None,
                "status": f"failed_{type(e).__name__}"
            })

    df_results = pd.DataFrame(results)
    print("\n=== Results ===")
    print(df_results.sort_values(by="coherence", ascending=False))

    # Show summary of failures
    failed = df_results[df_results['status'] != 'success']
    if len(failed) > 0:
        print(f"\n=== Failed Combinations: {len(failed)}/{len(df_results)} ===")
        print(failed[['min_cluster_size', 'n_neighbors', 'status']])

if __name__ == '__main__':
    main()