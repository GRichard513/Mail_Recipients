import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import time

cachedStopWords = stopwords.words("english")

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
        self.tfidf=TfidfVectorizer(tokenizer=None, stop_words='english')

    def fit(self, X):
        for i in range(X.shape[0]):
            text = X.body[i]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')
            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])
            self.token_dict[i] = y

        self.tfidf.fit(self.token_dict.values())


    def fit_transform(self, X):
        start_time = time.time()
        for i in range(X.shape[0]):
            text = X.body[i]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')

            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])

            self.token_dict[i] = y

        X_tfidf = self.tfidf.fit_transform(self.token_dict.values())

        print('performed Tf-Idf in %2i seconds.' % (time.time() - start_time))
        return X_tfidf

    def transform(self, Y):
        start_time = time.time()
        Y_dict={}
        for i in range(Y.shape[0]):
            text = Y.body[i]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')

            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])

            Y_dict[i] = y
        Y_tf_idf=self.tfidf.transform(Y_dict.values())

        print('performed Tf-Idf in %2i seconds.' % (time.time() - start_time))
        return Y_tf_idf
