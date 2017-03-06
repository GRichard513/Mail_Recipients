from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def complete_prediction(k, sender, address_books, res_temp, K=10):
    # k the number of recipients to predict
    k_most = [elt[0] for elt in address_books[sender][:K] if elt not in res_temp]
    k_most = k_most[:k]
    if len(k_most) < k: # sender n'a pas assez de contacts
        k_most.extend([0] * (k-len(k_most)))
    return k_most

class Predictor_1():
    def __init__(self, X, y, sender, address_books,N=10):
        self.train = X
        self.predict = y.values
        self.sender = sender
        self.N = min(N,10)
        self.address_books = address_books

    def predict_1(self, X):
        res = []#np.empty((X.shape[0],10))
        for i in range(X.shape[0]):
            cos = cosine_similarity(X[i],self.train).argsort()[:,30][0] # 30 mails les plus proches
            if self.N != 0:
                NN_recpt = {}
                for i in range(30):
                    for j in range self.predict[cos[i]]:
                        if self.predict[cos[i][j]] in NN_recpt:
                            NN_recpt[self.predict[cos[i][j]]] = 1
                        else:
                            NN_recpt[self.predict[cos[i][j]]]+= 1
                res_temp = NN_recpt.most_common(10)
                #res_temp = [self.predict[cos][0][:self.N]] # add the N first recipients of the closest e-mail
            else:
                 res_temp = []
            # if less than 10 recipients, complete the prediction with more frequents users
            if len(res_temp) < 10:
                res_temp.extend(complete_prediction(10-len(res_temp),self.sender, self.address_books, res_temp))
            res.append(res_temp)
        return res
