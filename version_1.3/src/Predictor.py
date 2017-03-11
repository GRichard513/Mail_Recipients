from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def complete_prediction(k, sender, address_books, res_temp, K=10):
    # k the number of recipients to predict
    k_most = [elt[0] for elt in address_books[sender][:K] if elt[0] not in res_temp]
    k_most = k_most[:k]
    if len(k_most) < k: # sender n'a pas assez de contacts
        k_most.extend([0] * (k-len(k_most)))
    return k_most

## Predictor closest message with Tf-Idf
class Predictor_TFIDF():
    def __init__(self, X, y, sender, address_books,N=10):
        self.train = X
        self.predict = y.values
        self.sender = sender
        self.address_books = address_books
        self.N = min(N,10)

    def prediction(self, X):
        res = []
        for i in range(X.shape[0]):
            cos = cosine_similarity(X[i],self.train).argsort()[:,0][0] # mail le plus proche
            if self.N != 0:
                res_temp = [self.predict[cos][0][:self.N]] # add the N first recipients of the closest e-mail
            else:
                 res_temp = []
            # if less than 10 recipients, complete the prediction with more frequents users
            if len(res_temp) < 10:
                res_temp.extend(complete_prediction(10-len(res_temp),self.sender, self.address_books, res_temp))
            res.append(res_temp)
        return res

## KNN
class Predictor_KNN():
    def __init__(self, X, y, sender, address_books,N=10):
        self.train = X
        self.predict = y.values
        self.sender = sender
        self.N = min(N,10)
        self.address_books = address_books

    def prediction(self, X):
        res = []
        for i in range(X.shape[0]): # loop trough each mails
            cos = (-cosine_similarity(X[i],self.train)).argsort()[:,:30][0] # 30 closest mails
            if self.N != 0:
                NN_recpt = {}
                for j in range(len(cos)): # len(cos) = 30 except if not enough mails, loop trough 30 closest mails
                    for k in range(len(self.predict[cos[j]])): # recipients in mails
                        if self.predict[cos[j]][k] in NN_recpt: # add recipient score to dictionnary
                            NN_recpt[self.predict[cos[j]][k]]+= cos[j]
                        else:
                            NN_recpt[self.predict[cos[j]][k]] = cos[j]
                res_temp = [k for k in sorted(NN_recpt, key=NN_recpt.get, reverse=True)][:10] # output 10 largest recipients scores
            else:
                 res_temp = []
            # if less than 10 recipients, complete the prediction with more frequents users
            if len(res_temp) < 10:
                res_temp.extend(complete_prediction(10-len(res_temp),self.sender, self.address_books, res_temp))
            res.append(res_temp)
        return res

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

        # perform centroid Tf-Idf. i.e each 10 most frequent recipients is represented
        # by an average of all mail he received.
        # exctract 10 most frequents recipients
        self.k_most = [elt[0] for elt in address_books[self.sender][:25]] # 20 more frequent recipients
        # perform average Tf-Idf on 10 most frequents recipients
        for recpt in self.k_most: # loop trough 10 most frequents recipients
            for i in range(X.shape[0]): # loop trough all mails send by sender
                if recpt in self.predict[i]: # if recipients is in mail
                    if recpt in self.X_recpt:
                        self.X_recpt[recpt] += X[i,:]
                    else:
                        self.X_recpt[recpt] = X[i,:]
            #self.X_recpt[recpt] = normalize(self.X_recpt[recpt], norm='l2', axis=1) # normalize tfidf vector

    def prediction(self, X):
        res = []
        cos = {}
        for i in range(X.shape[0]):
            # cosine similarity with 10 most frequents recpt
            for recpt, value in self.X_recpt.items():
                cos[recpt] = cosine_similarity(X[i],self.X_recpt[recpt])
            if self.N != 0:
                # return the 10 most frequent recipients in order
                # given by similarity to their centroid Tf-Idf representation
                res_temp = [k for k in sorted(cos, key=cos.get, reverse=True)]
            else:
                 res_temp = []
            # if less than 10 recipients, complete the prediction with more frequents users
            if len(res_temp) < 10:
                res_temp.extend(complete_prediction(10-len(res_temp),self.sender, self.address_books, res_temp))
            res.append(res_temp[:10])
        return res
