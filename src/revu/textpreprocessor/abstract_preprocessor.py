import re
import pandas as pd
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def preprocess_abstract_df(df):
    df = combine_text_columns(df, title_col="TI", abstract_col="AB", new_col="text")

    df["processed_text"] = df["text"].apply(
        lambda text: preprocess_text(text, custom_stop_words, nlp)
    )
    df["processed_text"] = remove_ubiquitous_words(df["processed_text"], threshold=0.95)

    return df

# Define custom stopwords
custom_stop_words = {
    "data", "analysis", "datum", "score", "method", "risk", "model", "effect", "result",
    "significant", "use", "can", "association", "year", "ci", "background", "objective", "aim", 
    "purpose", "introduction", "study", "measurement", "design", "variable", "test",
    'largely', 'substantially', 'later', 'es', 'fourth', 'end', 'primarily', 
    'medium', 'eg', 'respond', 'initially', 'lose', 'restrict', 'begin', 'second',
    'subsequent', 'free', 'immediate', 'close', 'regard', 'like', 'single', 
    'january', 'december', 'march', 'june', 'july', 'april', 'october', 
    'september', 'february', 'november', 'have', 'value', 'initial', 'common', 
    'incidence', 'occur', 'presence', 'month', 'measure', 'scale', 'time', 'day', 
    'approach', 'base', 'conclusion'
}

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
    """Remove short and non-alphabetic tokens."""
    return [token for token in tokens if len(token) > 1 and token.isalpha()]

def combine_text_columns(df, title_col="TI", abstract_col="AB", new_col="text"):
    if title_col in df.columns and abstract_col in df.columns:
        df[new_col] = df[title_col].fillna("") + " " + df[abstract_col].fillna("")
    else:
        raise KeyError(f"Missing columns: {title_col} and/or {abstract_col}")
    return df

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
