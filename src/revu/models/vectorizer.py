from sklearn.feature_extraction.text import CountVectorizer

def get_vectorizer_model(ngram_range=(1, 2), stop_words="english"):
    return CountVectorizer(min_df=1, max_df=1.0, ngram_range=ngram_range, stop_words=stop_words)
