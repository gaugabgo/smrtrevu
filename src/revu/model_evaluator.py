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
import argparse
import os
from datetime import datetime

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate BERTopic hyperparameters using pre-computed embeddings')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Input CSV with processed_text column')
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Path to pre-computed embeddings (.npy file). If not provided, will compute embeddings.')
    parser.add_argument('--model_name', type=str, default="all-MiniLM-L6-v2",
                        help='Embedding model name (only used if embeddings_path not provided)')
    parser.add_argument('--output_dir', type=str, default="hyperparameter_results",
                        help='Directory to save results')
    parser.add_argument('--text_column', type=str, default='processed_text',
                        help='Column name containing text')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample N documents for testing (useful for large datasets). If not set, uses all documents.')
    parser.add_argument('--low_memory', action='store_true',
                        help='Use low-memory UMAP settings (slower but uses less RAM)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load texts
    logger.info(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv, low_memory=False)

    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in CSV. Available columns: {list(df.columns)}")

    texts_for_bertopic = df[args.text_column].astype(str).tolist()
    logger.info(f"✓ Loaded {len(texts_for_bertopic)} documents")

    # Apply sampling if requested
    if args.sample_size and args.sample_size < len(texts_for_bertopic):
        logger.info(f"Sampling {args.sample_size} documents from {len(texts_for_bertopic)} total documents...")
        sample_indices = np.random.choice(len(texts_for_bertopic), size=args.sample_size, replace=False)
        sample_indices = sorted(sample_indices)  # Keep order for embeddings alignment
        texts_for_bertopic = [texts_for_bertopic[i] for i in sample_indices]
        logger.info(f"✓ Using {len(texts_for_bertopic)} sampled documents")

        # We'll also need to sample embeddings later if they're pre-computed
        sampled_embeddings_indices = sample_indices
    else:
        sampled_embeddings_indices = None

    # Handle embeddings
    if args.embeddings_path and os.path.exists(args.embeddings_path):
        logger.info(f"Loading pre-computed embeddings from {args.embeddings_path}...")
        embeddings = np.load(args.embeddings_path)
        logger.info(f"✓ Loaded embeddings with shape: {embeddings.shape}")

        # Sample embeddings if we sampled documents
        if sampled_embeddings_indices is not None:
            logger.info(f"Sampling embeddings to match sampled documents...")
            embeddings = embeddings[sampled_embeddings_indices]
            logger.info(f"✓ Sampled embeddings shape: {embeddings.shape}")

        # Validate
        if len(embeddings) != len(texts_for_bertopic):
            raise ValueError(f"Embeddings count ({len(embeddings)}) doesn't match document count ({len(texts_for_bertopic)})")

        embedding_model = args.model_name  # Use string when using pre-computed embeddings
    else:
        logger.info(f"Computing embeddings with model: {args.model_name}")
        embedding_model = SentenceTransformer(args.model_name)
        embeddings = embedding_model.encode(
            texts_for_bertopic,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        logger.info(f"✓ Embeddings computed with shape: {embeddings.shape}")

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

    # Define parameter grid for evaluation
    param_grid = ParameterGrid({
        "min_cluster_size": [15, 50, 100],
        "n_neighbors": [5, 15, 30],
        "n_components": [5, 10]
    })

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting hyperparameter search with {len(param_grid)} combinations")
    logger.info(f"{'='*60}\n")

    results = []
    total_combinations = len(param_grid)

    for idx, params in enumerate(param_grid, 1):

        min_cluster = params["min_cluster_size"]
        neighbors = params["n_neighbors"]
        n_comp = params["n_components"]

        logger.info(f"[{idx}/{total_combinations}] Testing: min_cluster_size={min_cluster}, n_neighbors={neighbors}, n_components={n_comp}")

        # Configure UMAP with low-memory settings if requested
        if args.low_memory:
            umap_model = UMAP(
                n_neighbors=neighbors,
                n_components=n_comp,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
                low_memory=True,
                n_jobs=1  # Single-threaded to reduce memory overhead
            )
        else:
            umap_model = UMAP(
                n_neighbors=neighbors,
                n_components=n_comp,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
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

            # Calculate additional metrics
            num_topics = len(topic_words)
            num_outliers = sum(1 for t in topics if t == -1)
            outlier_pct = (num_outliers / len(topics)) * 100

            logger.info(f"  ✓ Topics: {num_topics}, Outliers: {num_outliers} ({outlier_pct:.1f}%)")

            # Calculate coherence
            cm = CoherenceModel(
                topics=topic_words,
                texts=texts_tokenized,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence = cm.get_coherence()
            logger.info(f"  ✓ Coherence: {coherence:.4f}")

            results.append({
                "min_cluster_size": min_cluster,
                "n_neighbors": neighbors,
                "n_components": n_comp,
                "coherence": coherence,
                "num_topics": num_topics,
                "num_outliers": num_outliers,
                "outlier_pct": outlier_pct,
                "status": "success"
            })
            
        except ValueError as e:
            if "max_df corresponds to < documents than min_df" in str(e):
                logger.warning(f"✗ Params {params}: Clusters too small for vectorization. Skipping.")
                results.append({
                    "min_cluster_size": min_cluster,
                    "n_neighbors": neighbors,
                    "n_components": n_comp,
                    "coherence": None,
                    "num_topics": None,
                    "num_outliers": None,
                    "outlier_pct": None,
                    "status": "failed_small_clusters"
                })
            else:
                logger.error(f"  ✗ Unexpected ValueError - {e}")
                results.append({
                    "min_cluster_size": min_cluster,
                    "n_neighbors": neighbors,
                    "n_components": n_comp,
                    "coherence": None,
                    "num_topics": None,
                    "num_outliers": None,
                    "outlier_pct": None,
                    "status": f"failed_other: {str(e)}"
                })
        except Exception as e:
            logger.error(f"  ✗ Unexpected error - {type(e).__name__}: {e}")
            results.append({
                "min_cluster_size": min_cluster,
                "n_neighbors": neighbors,
                "n_components": n_comp,
                "coherence": None,
                "num_topics": None,
                "num_outliers": None,
                "outlier_pct": None,
                "status": f"failed_{type(e).__name__}"
            })

    # Create results DataFrame and save
    df_results = pd.DataFrame(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"hyperparameter_results_{timestamp}.csv")
    df_results.to_csv(results_path, index=False)
    logger.info(f"\n✓ Results saved to {results_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)

    successful = df_results[df_results['status'] == 'success']
    if len(successful) > 0:
        logger.info(f"\nSuccessful combinations: {len(successful)}/{len(df_results)}")
        logger.info("\nTop 5 by coherence:")
        top_5 = successful.sort_values(by="coherence", ascending=False).head(5)
        for idx, row in top_5.iterrows():
            logger.info(f"  {row['coherence']:.4f} | "
                       f"min_cluster={row['min_cluster_size']}, "
                       f"n_neighbors={row['n_neighbors']}, "
                       f"n_components={row['n_components']} | "
                       f"Topics: {row['num_topics']}, "
                       f"Outliers: {row['outlier_pct']:.1f}%")

        # Save best model info
        best_params = successful.sort_values(by="coherence", ascending=False).iloc[0]
        best_params_path = os.path.join(args.output_dir, f"best_params_{timestamp}.txt")
        with open(best_params_path, 'w') as f:
            f.write(f"Best Hyperparameters (by coherence)\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"min_cluster_size: {int(best_params['min_cluster_size'])}\n")
            f.write(f"n_neighbors: {int(best_params['n_neighbors'])}\n")
            f.write(f"n_components: {int(best_params['n_components'])}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  Coherence: {best_params['coherence']:.4f}\n")
            f.write(f"  Number of topics: {int(best_params['num_topics'])}\n")
            f.write(f"  Outliers: {int(best_params['num_outliers'])} ({best_params['outlier_pct']:.1f}%)\n")
        logger.info(f"\n✓ Best parameters saved to {best_params_path}")

        # Print full results table
        logger.info(f"\nFull results table:\n{successful.sort_values(by='coherence', ascending=False).to_string(index=False)}")
    else:
        logger.warning("\nNo successful combinations found!")

    # Show summary of failures
    failed = df_results[df_results['status'] != 'success']
    if len(failed) > 0:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"Failed Combinations: {len(failed)}/{len(df_results)}")
        logger.warning(f"{'='*60}")
        logger.warning(f"\n{failed[['min_cluster_size', 'n_neighbors', 'n_components', 'status']].to_string(index=False)}")

    logger.info("\n" + "="*60)
    logger.info("✅ Hyperparameter evaluation complete!")
    logger.info("="*60)

if __name__ == '__main__':
    main()