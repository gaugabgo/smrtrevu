from revu.textpreprocessor.abstract_preprocessor import preprocess_abstract_df
from revu.textpreprocessor.ft_preprocessor import preprocess_fulltext

def run_preprocessing(data_source, source_type, **kwargs):
    """
    Dispatch to the appropriate preprocessing function.

    Parameters:
    - data_source: A DataFrame (for abstracts) or a folder path (for full text).
    - source_type: One of ["abstract", "fulltext"].
    - **kwargs: Additional parameters passed to the preprocessing function.
      For abstracts: chunk_size, n_workers, checkpoint_dir, threshold

    Returns:
    - Preprocessed output.
    """
    if source_type == "abstract":
        return preprocess_abstract_df(data_source, **kwargs)
    elif source_type == "fulltext":
        return preprocess_fulltext(data_source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")