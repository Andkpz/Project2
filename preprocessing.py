# preprocessing.py
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")

def preprocess_text(text):
    """Обработка одного текста (лемматизация/стемминг)"""
    if not isinstance(text, str):
        return ""
    
    tokens = word_tokenize(text, language='russian')
    processed = []
    
    for token in tokens:
        if token.isalpha() and token.lower() not in stop_words:
            processed.append(stemmer.stem(token.lower()))
    
    return " ".join(processed)