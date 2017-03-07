import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

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

    def fit_transform(self, X):
        for i in range(X.shape[0]):
            text = X.body[i]
            lowers = text.lower()
            s=string.punctuation.replace('@','')
            s=s.replace('+','')

            no_punctuation = lowers.translate(str.maketrans('','',s))
            y = " ".join(no_punctuation.split())
            y = ' '.join([word for word in y.split() if word not in cachedStopWords])

            self.token_dict[i] = y

        tfidf = TfidfVectorizer(tokenizer=None, stop_words='english')
        X_tfidf = tfidf.fit_transform(self.token_dict.values())

        return X_tfidf