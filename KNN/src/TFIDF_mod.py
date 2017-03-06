import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

class TFIDF():
    def __init__(self):
        self.token_dict = {}

    def fit_transform(self, X):
        for i in range(X.shape[0]):
            text = X.body[i]
            lowers = text.lower()
            no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
            self.token_dict[i] = no_punctuation

        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        X_tfidf = tfidf.fit_transform(self.token_dict.values())

        return X_tfidf
