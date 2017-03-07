import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import sys

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

class FeatureExtractor():
    def __init__(self):
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        self.token_dict = {}

    def fit(self, training_info, y):
        print("fit")
        for i in range(training_info.shape[0]):
            #progress(i, training_info.shape[0]-1, status='fit tfidf')
            text = training_info.body.values[i]
            lowers = text.lower()
            no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
            self.token_dict[i] = no_punctuation

        self.tfidf.fit(self.token_dict.values())
        #pass

    def transform(self, training_info):
        print('transform')
        return self.tfidf.transform(self.token_dict.values())
        #pass
