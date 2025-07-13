import re
import pandas as pd
import os
import spacy
from gensim.models.phrases import Phrases, Phraser

def preprocess_fulltext_txt(txt_dir):
    # Specific logic for txt files from PDFs
    return list_of_clean_texts

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define custom stopwords
custom_stop_words = {
    "data", "analysis", "datum", "score", "method", "risk", "data", "model", "effect", "result",
    "significant", "use", "can", "association", "year", "ci", "background", "objective", "aim", 
    "purpose", "introduction",  "data", "study", "datum", "measurement", "score", "analysis", 
    "design", "variable", "test", 'largely', 'substantially', 'later', 'es', 'fourth', 'end', 'primarily', 
    'medium', 'eg', 'respond', 'initially', 'lose', 'restrict', 'begin', 'second', 'subsequent', 'free', 'immediate',
    'close','regard', 'like', 'single', 'january', 'december', 'march', 'june', 'july', 'april', 'october', 
    'september', 'february', 'november', 'have','value', 'initial','common', 'incidence', 'occur','presence',
    'month', 'measure','scale','time', 'day','approach', 'base', 'year', 'method', 'conclusion', 'abstract', 'patient', 
    'study', 'group', 'result', 'year'
}

# Load allowed terms for filtering
def load_allowed_terms(filepath):
    """Load allowed terms from file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            return set(file.read().splitlines())
    except FileNotFoundError:
        print(f"Warning: Allowed terms file not found at {filepath}")
        return set()

# Update this path to your methodological terms file
allowed_terms = load_allowed_terms("/Users/gabriellegauthier/Desktop/Python Topic Model/TM_MethodologicalTerms_utf8.txt")

def clean_tokens(tokens):
    """Clean the tokens by removing non-alphabetic characters and short tokens."""
    cleaned_tokens = [token for token in tokens if len(token) > 1 and token.isalpha()]
    return cleaned_tokens

def load_text_files_corpus(text_directory):
    """
    Load text files from directory and create a corpus DataFrame.
    Assumes text files were created by the PDF extraction tool.
    """
    text_data = []
    
    # Get all text files in the directory
    text_files = [f for f in os.listdir(text_directory) if f.lower().endswith('.txt')]
    
    print(f"Found {len(text_files)} text files in {text_directory}")
    
    for i, filename in enumerate(text_files):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing file {i+1}/{len(text_files)}: {filename}")
        
        text_path = os.path.join(text_directory, filename)
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the header information if present
            lines = content.split('\n')
            original_filename = ""
            extraction_method = ""
            total_pages = 0
            
            # Extract metadata from header
            for line in lines[:10]:  # Check first 10 lines for metadata
                if line.startswith("FILENAME:"):
                    original_filename = line.replace("FILENAME:", "").strip()
                elif line.startswith("EXTRACTION METHOD:"):
                    extraction_method = line.replace("EXTRACTION METHOD:", "").strip()
                elif line.startswith("TOTAL PAGES:"):
                    try:
                        total_pages = int(line.replace("TOTAL PAGES:", "").strip())
                    except:
                        total_pages = 0
            
            # Find the start of actual text content (after the separator line)
            text_start_idx = 0
            for j, line in enumerate(lines):
                if "=" * 20 in line:  # Look for separator line
                    text_start_idx = j + 1
                    break
            
            # Extract the main text content
            if text_start_idx > 0:
                main_text = '\n'.join(lines[text_start_idx:])
            else:
                main_text = content  # Fallback if no separator found
            
            # Clean up common PDF extraction artifacts
            main_text = clean_pdf_artifacts(main_text)
            
            text_data.append({
                'text_filename': filename,
                'original_pdf_filename': original_filename if original_filename else filename.replace('.txt', '.pdf'),
                'extraction_method': extraction_method,
                'total_pages': total_pages,
                'text': main_text,
                'text_length': len(main_text),
                'word_count': len(main_text.split()) if main_text else 0
            })
            
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            # Add empty entry for failed files
            text_data.append({
                'text_filename': filename,
                'original_pdf_filename': filename.replace('.txt', '.pdf'),
                'extraction_method': 'error',
                'total_pages': 0,
                'text': '',
                'text_length': 0,
                'word_count': 0
            })
    
    df = pd.DataFrame(text_data)
    
    print(f"\nLoaded {len(df)} text documents")
    print(f"Average text length: {df['text_length'].mean():.0f} characters")
    print(f"Average word count: {df['word_count'].mean():.0f} words")
    print(f"Documents with no text: {sum(df['text_length'] == 0)}")
    print(f"Documents with < 100 words: {sum(df['word_count'] < 100)}")
    
    return df

def clean_pdf_artifacts(text):
    """Clean common PDF extraction artifacts."""
    if not text:
        return ""
    
    # Remove page markers
    text = re.sub(r'\n--- PAGE \d+ ---\n', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    
    # Remove common PDF artifacts
    text = re.sub(r'\f', ' ', text)  # Form feed characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control characters
    
    # Remove isolated single characters that are likely artifacts
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    
    # Clean up remaining whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Train phrase models
def train_phrases_model(df, text_column="text"):
    """Train bigram and trigram models from text data."""
    print("Training phrase models...")
    
    # Filter out empty texts and very short texts
    valid_texts = df[df[text_column].str.len() > 100][text_column].dropna()
    print(f"Using {len(valid_texts)} documents for phrase model training")
    
    if len(valid_texts) == 0:
        print("Warning: No valid texts found for phrase model training!")
        return None, None
    
    # Tokenize texts for phrase training
    tokenized_texts = []
    for text in valid_texts:
        # Basic tokenization for phrase model training
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        tokenized_texts.append(tokens)
    
    print(f"Average tokens per document: {sum(len(t) for t in tokenized_texts) / len(tokenized_texts):.0f}")
    
    bigram = Phrases(tokenized_texts, min_count=5, threshold=10)
    trigram = Phrases(bigram[tokenized_texts], min_count=3, threshold=5)
    
    print("Phrase models trained successfully")
    return Phraser(bigram), Phraser(trigram)

# Initialize global counter
no_tokens_count = 0

# Workflow 1: Preprocessing WITHOUT Filtering
def preprocess_no_filtering(text):
    global no_tokens_count

    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 50:
        no_tokens_count += 1
        return ""

    # Remove URLs and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    # Process with spaCy
    try:
        doc = nlp(text[:1000000])  # Limit text length for spaCy processing
    except:
        no_tokens_count += 1
        return ""
    
    all_stop_words = nlp.Defaults.stop_words.union(custom_stop_words)

    # Tokenization and cleanup
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space and token.is_alpha]
    tokens = clean_tokens(tokens)

    # Remove stopwords BEFORE applying bigram/trigram models
    tokens = [token for token in tokens if token not in all_stop_words and len(token) > 2]

    if not tokens or len(tokens) < 5:
        no_tokens_count += 1
        return ""

    # Apply phrase models AFTER stopword removal
    if bigram_model and trigram_model:
        bigram_tokens = bigram_model[tokens]
        trigram_tokens = trigram_model[bigram_tokens]
        final_tokens = trigram_tokens
    else:
        final_tokens = tokens

    # Lemmatize the final tokens
    try:
        lemmatized_text = " ".join(final_tokens)
        lemma_doc = nlp(lemmatized_text[:1000000])
        lemmatized_tokens = [token.lemma_ for token in lemma_doc if token.is_alpha and len(token.lemma_) > 2]
        return " ".join(lemmatized_tokens)
    except:
        return " ".join(final_tokens)

# Workflow #2: Filter by list of key terms
secondary_stopwords = {
    "data", "study", "datum", "measurement", "score", "analysis", "design", "variable", "test"
}

def preprocess_with_filtering(text):
    global no_tokens_count

    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 50:
        no_tokens_count += 1
        return ""

    # Remove URLs and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    # Process with spaCy
    try:
        doc = nlp(text[:1000000])  # Limit text length for spaCy processing
    except:
        no_tokens_count += 1
        return ""

    # Apply dictionary filter and then secondary stopword removal
    filtered_tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
        and len(token.lemma_) > 2
        and token.lemma_.lower() in allowed_terms
        and token.lemma_.lower() not in secondary_stopwords
    ]

    if not filtered_tokens or len(filtered_tokens) < 5:
        no_tokens_count += 1
        return ""

    # Apply phrase models
    if bigram_model and trigram_model:
        bigram_tokens = bigram_model[filtered_tokens]
        trigram_tokens = trigram_model[bigram_tokens]
        return " ".join(trigram_tokens)
    else:
        return " ".join(filtered_tokens)