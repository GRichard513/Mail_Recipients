import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import time
import warnings
warnings.filterwarnings("ignore")

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

    def fit(self, X_df):
        X = X_df.values
        for i in range(X.shape[0]):
            text = X[i,2]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')
            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])
            self.token_dict[i] = y

        self.tfidf.fit(self.token_dict.values())


    def fit_transform(self, X_df):
        X = X_df.values
        for i in range(X.shape[0]):
            text = X[i,2]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')

            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])

            self.token_dict[i] = y

        X_tfidf = self.tfidf.fit_transform(self.token_dict.values())

        return X_tfidf

    def transform(self, Y_df):
        Y = Y_df.values
        Y_dict={}
        for i in range(Y.shape[0]):
            text = Y[i,2]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')

            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])

            Y_dict[i] = y
        Y_tf_idf=self.tfidf.transform(Y_dict.values())

        return Y_tf_idf

class LDA():
    def __init__(self):
        self.n_features = 1000
        self.n_topics = 10
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english')
        self.lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)

    def fit_transform(self, X_df):
        X = X_df.body.values
        tf = tf_vectorizer.fit_transform(X)
        X_LDA = self.lda.fit_transform(tf)
        return X_LDA

    def transform(self, X_df):
        X = X_df.body.values
        tf = tf_vectorizer.transform(X)
        X_LDA = self.lda.transform(tf)
        return X_LDA
