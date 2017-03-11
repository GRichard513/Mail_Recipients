from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def softmax(X):
    exp = np.exp(X)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def complete_prediction(k, sender, address_books, res_, K=10):
    # k the number of recipients to predict
    k_most = [elt[0] for elt in address_books[sender][:K] if elt[0] not in res_]
    k_most = k_most[:k]
    if len(k_most) < k: # sender n'a pas assez de contacts
        k_most.extend([0] * (k-len(k_most)))
    return k_most

## KNN
class Predictor_KNN():
    def __init__(self, X, y, sender, address_books,N=10):
        self.train = X
        self.predict = y.values
        self.sender = sender
        self.N = min(N,10)
        self.address_books = address_books
        self.res = []
        self.proba = []

    def prediction(self, X):
        for i in range(X.shape[0]): # loop trough each mails
            cos = cosine_similarity(X[i],self.train)[0] # cosine similarity
            close = (-cosine_similarity(X[i],self.train)).argsort()[:,:][0] # 30 closest mails
            if self.N != 0:
                NN_recpt = {}
                for j in range(len(close)): # len(close) = 30 except if not enough mails, loop trough 30 closest mails
                    for k in range(len(self.predict[close[j]])): # recipients in mails
                        if self.predict[close[j]][k] in NN_recpt: # add recipient score to dictionnary
                            NN_recpt[self.predict[close[j]][k]]+= cos[close[j]]
                        else:
                            NN_recpt[self.predict[close[j]][k]] = cos[close[j]]
                res_ = sorted(NN_recpt, key=NN_recpt.get, reverse=True) # output 10 largest recipients scores
                proba_ = sorted(NN_recpt.values(), reverse=True)
            else:
                 res_ = []
                 proba_ = [0.1]*10 # return uniform proba
            # if less than 10 recipients, complete the prediction with more frequents users
            if len(res_) < 10:
                res_.extend(complete_prediction(10-len(res_),self.sender, self.address_books, res_))
                if self.N != 0:
                    self.proba.extend([min(cos[:self.N])]*(10-len(res_))) # if not enough recipients complete proba with min proba uniform
            self.res.append(res_[:10])
            self.proba.append(proba_[:10])
        return self.res

    def prediction_proba(self):
        return softmax(self.proba)

import scipy

## Predictor Tf-Idf centralized centroïd

class Predictor_CTFIDF():
    def __init__(self, X, y, sender, address_books,N=10):
        self.N = min(N,10)
        self.train = X
        self.predict = y.values
        self.sender = sender
        self.address_books = address_books
        self.X_recpt = {}
        self.res = []
        self.proba = []

        # perform centroid Tf-Idf. i.e each 10 most frequent recipients is represented
        # by an average of all mail he received.
        # exctract 10 most frequents recipients
        self.k_most = [elt[0] for elt in address_books[self.sender][:20]] # 20 more frequent recipients
        # perform average Tf-Idf on 10 most frequents recipients
        for recpt in self.k_most: # loop trough 10 most frequents recipients
            for i in range(X.shape[0]): # loop trough all mails send by sender
                if recpt in self.predict[i]: # if recipients is in mail
                    if recpt in self.X_recpt:
                        self.X_recpt[recpt] += X[i,:]
                    else:
                        self.X_recpt[recpt] = X[i,:]

    def prediction(self, X):
        cos = {}
        for i in range(X.shape[0]):
            # cosine similarity with 10 most frequents recpt
            for recpt, value in self.X_recpt.items():
                cos[recpt] = cosine_similarity(X[i],self.X_recpt[recpt])[0,0]
            if self.N != 0:
                # return the 10 most frequent recipients in order
                # given by similarity to their centroid Tf-Idf representation
                res_ = sorted(cos, key=cos.get, reverse=True)
                proba_ = sorted(cos.values(), reverse=True)
            else:
                 res_ = []
                 proba_ = [0.1]*10 # return uniform proba
            # if less than 10 recipients, complete the prediction with more frequents users
            if len(res_) < 10:
                res_.extend(complete_prediction(10-len(res_),self.sender, self.address_books, res_))
                if self.N != 0:
                    proba_.extend([0.1]*(10-len(res_))) # if not enough recipients complete proba with min proba uniform
            self.res.append(res_[:10])
            self.proba.append(proba_[:10])
        return self.res

    def prediction_proba(self):
        return softmax(self.proba)

# ## Predictor closest message with Tf-Idf
# class Predictor_TFIDF():
#     def __init__(self, X, y, sender, address_books,N=10):
#         self.train = X
#         self.predict = y.values
#         self.sender = sender
#         self.address_books = address_books
#         self.N = min(N,10)
#         self.res = []
#         self.proba = []
#
#     def prediction(self, X):
#         for i in range(X.shape[0]):
#
#             cos = cosine_similarity(X[i],self.train).argsort()[:,0][0] # mail le plus proche
#             print(cos)
#             if self.N != 0:
#                 res_ = [self.predict[cos][0][:self.N]] # add the N first recipients of the closest e-mail
#                 proba_ = cos[:self.N] # proba is identified with cosine distance
#             else:
#                  res_ = []
#                  proba_ = [0.1]*10
#             # if less than 10 recipients, complete the prediction with more frequents users
#             if len(res_) < 10:
#                 res_.extend(complete_prediction(10-len(res_),self.sender, self.address_books, res_))
#                 if self.N != 0:
#                     self.proba.extend([min(cos[:self.N])]*(10-len(res_))) # if not enough recipients complete proba with min proba uniform
#             self.res.append(res_[:10])
#             self.proba.append(proba_[:10])
#         return self.res
#
#     def prediction_proba(self):
#         return softmax(np.array(self.proba))
