from textpreprocessor.abstract_preprocessor import preprocess_abstract_df
from textpreprocessor.ft_preprocessor import preprocess_fulltext_txt

def run_preprocessing(data_source, source_type):
    """
    Dispatch to the appropriate preprocessing function.

    Parameters:
    - data_source: A DataFrame (for abstracts) or a folder path (for full text).
    - source_type: One of ["abstract", "fulltext"].

    Returns:
    - Preprocessed output.
    """
    if source_type == "abstract":
        return preprocess_abstract_df(data_source)
    elif source_type == "fulltext":
        return preprocess_fulltext_txt(data_source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")