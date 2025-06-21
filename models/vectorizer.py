from sklearn.feature_extraction.text import CountVectorizer

def get_vectorizer_model(ngram_range=(1, 2), stop_words="english"):
    return CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
