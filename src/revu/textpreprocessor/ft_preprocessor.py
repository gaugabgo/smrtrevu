import re
import pandas as pd
import spacy
from gensim.models.phrases import Phrases, Phraser

# Load SpaCy model 
nlp = spacy.load("en_core_web_sm")

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

def preprocess_sections(text, stop_words, nlp_model):
    """Clean and tokenize text - returns intermediate processed text WITHOUT phrase models."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # XML entities
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)
    text = re.sub(r'&nbsp;', ' ', text)
    
    # Remove remaining XML tags 
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r';', ' | ', text) 
    text = re.sub(r'["""]', '"', text)  
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    doc = nlp_model(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    tokens = clean_tokens(tokens)
    tokens = [token for token in tokens if token not in stop_words]

    if not tokens:
        return ""

    return " ".join(tokens)

# Concatenate headers and section texts in df
def combine_unique_headers(headers_series):
    """Combine unique non-null headers with semicolon."""
    unique_headers = headers_series.dropna().unique()
    unique_headers = [h for h in unique_headers if h.strip()]
    return '; '.join(unique_headers) if unique_headers else ''

def combine_text_content(content_series):
    """Combine all content with semicolon, filtering out empty content."""
    content_list = content_series.dropna().tolist()
    content_list = [c.strip() for c in content_list if c.strip()]
    return '; '.join(content_list) if content_list else ''

def combine_unique_values(values_series):
    """Combine unique values with semicolon."""
    return '; '.join(values_series.unique())

def consolidate_by_file(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate DataFrame to one row per file."""
    if df.empty:
        print("No data to consolidate.")
        return pd.DataFrame()
    
    print(f"Consolidating {len(df)} rows into one row per file...")
    
    consolidated = df.groupby('filename').agg({
        'title': 'first',
        'pub_date': 'first',
        'authors': 'first', 
        'parent_header': combine_unique_headers,
        'content_type': combine_unique_values,
        'content': combine_text_content
    }).reset_index()
    
    consolidated.rename(columns={
        'parent_header': 'all_headers',
        'content_type': 'content_types_found', 
        'content': 'combined_content'
    }, inplace=True)
    
    print(f"Consolidated to {len(consolidated)} rows (one per file)")
    return consolidated

def load_and_consolidate_csv(csv_file_path: str) -> pd.DataFrame:
    """Load existing CSV file and consolidate to one row per file."""
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from {csv_file_path}")
        consolidated_df = consolidate_by_file(df)  
        return consolidated_df
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return pd.DataFrame()
    
def train_phrases_model(texts):
    """Train bigram and trigram phrase models."""
    valid_texts = [text for text in texts.dropna() if text.strip()]
    tokenized_texts = [text.split() for text in valid_texts]
    
    if not tokenized_texts:
        print("No valid texts for phrase model training")
        return None, None
    
    print(f"Training phrase models on {len(tokenized_texts)} texts...")
    bigram = Phrases(tokenized_texts, min_count=5, threshold=10)
    trigram = Phrases(bigram[tokenized_texts], min_count=3, threshold=5)
    return Phraser(bigram), Phraser(trigram)

def apply_phrases_model(text, bigram_model, trigram_model, nlp_model):
    """Apply phrase models and lemmatization to preprocessed text."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    
    if bigram_model is None or trigram_model is None:
        print("Warning: Phrase models not available, applying lemmatization only")
        final_doc = nlp_model(text)
        return " ".join([token.lemma_ for token in final_doc if not token.is_punct and not token.is_space])

    tokens = text.split()
    bigram_tokens = bigram_model[tokens]
    trigram_tokens = trigram_model[bigram_tokens]

    final_doc = nlp_model(" ".join(trigram_tokens))
    return " ".join([token.lemma_ for token in final_doc if not token.is_punct and not token.is_space])

def preprocess_fulltext(df):
    """Main orchestrator function for preprocessing workflow."""
    print("Starting text preprocessing workflow...")
    
    print("Step 1: Cleaning and tokenizing text...")
    df["content"] = df["content"].apply(
        lambda text: preprocess_sections(text, custom_stop_words, nlp)
    )
    
    print("Step 2: Consolidating sections by file...")
    df_consolidated = consolidate_by_file(df)
    
    print("Step 3: Training phrase models...")
    bigram_model, trigram_model = train_phrases_model(df_consolidated["combined_content"])
    
    print("Step 4: Applying phrase models and lemmatization...")
    df_consolidated["processed_text"] = df_consolidated["combined_content"].apply(
        lambda text: apply_phrases_model(text, bigram_model, trigram_model, nlp)
    )
    
    print("Text preprocessing workflow completed!")
    print("Text length stats:")
    print(df_consolidated["processed_text"].str.len().describe())
    return df_consolidated

if __name__ == "__main__":
    df = pd.read_csv("tei_extracted_data_20250908_181903.csv")
    
    result = preprocess_fulltext(df)
    
    result.to_csv("test_ft_preprocess.csv", index=False)