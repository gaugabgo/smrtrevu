import re
import pandas as pd
import spacy
from gensim.models.phrases import Phrases, Phraser

def preprocess_abstract_df(df):
    # Specific logic for abstracts
    return df

# Load SpaCy model once at the module level
nlp = spacy.load("en_core_web_sm")

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

def clean_tokens(tokens):
    """Remove short and non-alphabetic tokens."""
    return [token for token in tokens if len(token) > 1 and token.isalpha()]

def combine_text_columns(df, title_col="TI", abstract_col="AB", new_col="text"):
    if title_col in df.columns and abstract_col in df.columns:
        df[new_col] = df[title_col].fillna("") + " " + df[abstract_col].fillna("")
    else:
        raise KeyError(f"Missing columns: {title_col} and/or {abstract_col}")
    return df

def train_phrases_model(texts):
    """Train bigram and trigram phrase models."""
    tokenized_texts = [text.split() for text in texts.dropna()]
    bigram = Phrases(tokenized_texts, min_count=5, threshold=10)
    trigram = Phrases(bigram[tokenized_texts], min_count=3, threshold=5)
    return Phraser(bigram), Phraser(trigram)

def preprocess_text(text, bigram_model, trigram_model, stop_words, nlp_model):
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

    bigram_tokens = bigram_model[tokens]
    trigram_tokens = trigram_model[bigram_tokens]

    final_doc = nlp_model(" ".join(trigram_tokens))
    return " ".join([token.lemma_ for token in final_doc])
