import click

from revu.model_evaluator import run_evaluation


@click.command()
@click.option("--input_csv", required=True, type=click.Path(exists=True),
              help="CSV file with processed text column.")
@click.option("--output_dir", default="hyperparameter_results", show_default=True,
              help="Directory to save results CSV and best-params file.")
@click.option("--embeddings_path", default=None, type=click.Path(),
              help="Pre-computed embeddings .npy file. If omitted, embeddings are computed on the fly.")
@click.option("--model_name", default="all-MiniLM-L6-v2", show_default=True,
              help="Sentence Transformers model name (used only if --embeddings_path is not provided).")
@click.option("--text_column", default="processed_text", show_default=True,
              help="Column in the input CSV containing text to model.")
@click.option("--sample_size", default=None, type=int,
              help="Evaluate on a random sample of N documents. Useful for large corpora.")
@click.option("--low_memory", is_flag=True, default=False,
              help="Use single-threaded, low-memory UMAP settings (slower).")
@click.option("--min-cluster-sizes", multiple=True, type=int, default=(100, 200, 300, 500),
              show_default=True,
              help="HDBSCAN min_cluster_size values to evaluate. Repeat to add values.")
@click.option("--n-neighbors-values", multiple=True, type=int, default=(15, 50),
              show_default=True,
              help="UMAP n_neighbors values to evaluate. Repeat to add values.")
@click.option("--n-components-values", multiple=True, type=int, default=(5, 15),
              show_default=True,
              help="UMAP n_components values to evaluate. Repeat to add values.")
def evaluate_model(
    input_csv, output_dir, embeddings_path, model_name, text_column,
    sample_size, low_memory, min_cluster_sizes, n_neighbors_values, n_components_values,
):
    """Evaluate BERTopic hyperparameters across a grid and rank by coherence (c_v).

    Fits a BERTopic model for every combination of --min-cluster-sizes,
    --n-neighbors-values, and --n-components-values, then saves a ranked results
    CSV and a best-params summary file.
    """
    run_evaluation(
        input_csv=input_csv,
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        model_name=model_name,
        text_column=text_column,
        sample_size=sample_size,
        low_memory=low_memory,
        min_cluster_sizes=list(min_cluster_sizes),
        n_neighbors_values=list(n_neighbors_values),
        n_components_values=list(n_components_values),
    )
