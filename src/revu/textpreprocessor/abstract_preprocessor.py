import re
import pandas as pd
import spacy
from collections import Counter
from multiprocessing import Pool, cpu_count
import pickle
from pathlib import Path
from typing import Optional, List
import numpy as np

nlp = spacy.load("en_core_web_sm")

def preprocess_abstract_df(
    df,
    chunk_size: int = 1000,
    n_workers: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    threshold: float = 0.95
):
    """
    Preprocess abstract DataFrame with chunking, parallelization, and checkpointing.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with abstract data
    chunk_size : int, default=1000
        Number of rows to process per chunk
    n_workers : int, optional
        Number of parallel workers (defaults to cpu_count - 1)
    checkpoint_dir : str, optional
        Directory to save checkpoints. If None, no checkpointing is used.
    threshold : float, default=0.95
        Threshold for removing ubiquitous words

    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    print(f"🚀 Starting preprocessing with chunking and parallelization...")
    print(f"   Total rows: {len(df)}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Workers: {n_workers if n_workers else cpu_count() - 1}")
    print(f"   Checkpointing: {'Enabled' if checkpoint_dir else 'Disabled'}")

    # Combine text columns
    df = combine_text_columns(df, title_col="title", abstract_col="abstract", new_col="text")

    # Setup checkpoint directory if enabled
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None

    # Split DataFrame into chunks
    num_chunks = int(np.ceil(len(df) / chunk_size))
    chunks = np.array_split(df, num_chunks)

    print(f"\n📦 Processing {num_chunks} chunks...")

    processed_chunks = []

    for idx, chunk in enumerate(chunks):
        print(f"\n⏳ Processing chunk {idx + 1}/{num_chunks} ({len(chunk)} rows)...")

        # Check for existing checkpoint
        if checkpoint_path:
            cached_chunk = load_checkpoint(checkpoint_path, idx)
            if cached_chunk is not None:
                print(f"✅ Loaded from checkpoint")
                processed_chunks.append(cached_chunk)
                continue

        # Process chunk in parallel
        texts = chunk["text"].tolist()
        processed_texts = process_chunk_parallel(texts, n_workers=n_workers)

        # Create processed chunk
        chunk_copy = chunk.copy()
        chunk_copy["processed_text"] = processed_texts

        # Save checkpoint
        if checkpoint_path:
            save_checkpoint(chunk_copy, checkpoint_path, idx)

        processed_chunks.append(chunk_copy)
        print(f"✅ Chunk {idx + 1}/{num_chunks} completed")

    # Combine all processed chunks
    print("\n🔗 Combining all chunks...")
    df_processed = pd.concat(processed_chunks, ignore_index=True)

    # Remove ubiquitous words across all documents
    print(f"\n🧹 Removing ubiquitous words (threshold: {threshold})...")
    df_processed["processed_text"] = remove_ubiquitous_words(
        df_processed["processed_text"], threshold=threshold
    )

    # Clean up checkpoints if successful
    if checkpoint_path:
        clean_checkpoints(checkpoint_path)

    print("\n✅ Preprocessing complete!")
    return df_processed

# Define custom stopwords
custom_stop_words = {
    'background', 'introduction', 'methods', 'methods', 'results', 'conclusion', 'nan'
}

# Checkpoint utility functions
def save_checkpoint(data, checkpoint_path: Path, chunk_idx: int):
    """Save checkpoint for resuming processing."""
    checkpoint_file = checkpoint_path / f"checkpoint_chunk_{chunk_idx}.pkl"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Checkpoint saved: {checkpoint_file}")

def load_checkpoint(checkpoint_path: Path, chunk_idx: int):
    """Load checkpoint if it exists."""
    checkpoint_file = checkpoint_path / f"checkpoint_chunk_{chunk_idx}.pkl"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def clean_checkpoints(checkpoint_path: Path):
    """Remove all checkpoint files after successful completion."""
    if checkpoint_path.exists():
        for file in checkpoint_path.glob("checkpoint_chunk_*.pkl"):
            file.unlink()
        print(f"🧹 Cleaned up checkpoints from {checkpoint_path}")

def remove_ubiquitous_words(processed_texts, threshold=0.95):
    """Remove words that appear in more than threshold% of documents."""
    # Count documents containing each word
    word_doc_count = Counter()
    total_docs = len(processed_texts)
    
    for text in processed_texts:
        if pd.isna(text) or not isinstance(text, str):
            continue
        # Get unique words in this document
        unique_words = set(text.split())
        word_doc_count.update(unique_words)
    
    # Identify ubiquitous words
    ubiquitous_words = {
        word for word, count in word_doc_count.items() 
        if count / total_docs >= threshold
    }
    
    print(f"Removing {len(ubiquitous_words)} ubiquitous words appearing in >={threshold*100}% of documents:")
    print(f"  {sorted(ubiquitous_words)}")
    
    filtered_texts = []
    for text in processed_texts:
        if pd.isna(text) or not isinstance(text, str):
            filtered_texts.append("")
        else:
            words = text.split()
            filtered_words = [word for word in words if word not in ubiquitous_words]
            filtered_texts.append(" ".join(filtered_words))
    
    return filtered_texts

def clean_tokens(tokens):
    """Remove short, non-alphabetic, and non-English (non-ASCII) tokens."""
    return [token for token in tokens if len(token) > 1 and token.isalpha() and token.isascii()]

def combine_text_columns(df, title_col="TI", abstract_col="AB", new_col="text"):
    if title_col in df.columns and abstract_col in df.columns:
        df[new_col] = df[title_col].fillna("") + " " + df[abstract_col].fillna("")
    else:
        raise KeyError(f"Missing columns: {title_col} and/or {abstract_col}")
    return df

def _preprocess_text_wrapper(text):
    """Wrapper function for parallel processing."""
    return preprocess_text(text, custom_stop_words, nlp)

def process_chunk_parallel(texts: List[str], n_workers: Optional[int] = None) -> List[str]:
    """Process a chunk of texts in parallel using multiprocessing."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one core free

    with Pool(processes=n_workers) as pool:
        processed_texts = pool.map(_preprocess_text_wrapper, texts)

    return processed_texts

def preprocess_text(text, stop_words, nlp_model):
    """Clean and tokenize text, apply phrase models, and lemmatize."""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    doc = nlp_model(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    tokens = clean_tokens(tokens)
    tokens = [token for token in tokens if token not in stop_words]

    if not tokens:
        return ""

    final_doc = nlp_model(" ".join(tokens))
    return " ".join([token.lemma_ for token in final_doc])
