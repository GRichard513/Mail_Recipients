import numpy as np
from proper_name_extractor import extract_names

def complete_prediction(k, sender, address_books, res_temp, K=10):
    # k the number of recipients to predict
    k_most = [elt[0] for elt in address_books[sender][:K] if elt not in res_temp]
    k_most = k_most[:k]
    if len(k_most) < k: # sender n'a pas assez de contacts
        k_most.extend([0] * (k-len(k_most)))
    return k_most

class Predictor_Names():
    def __init__(self, X, y, sender, address_books):
        self.train = X
        self.predict = y.values
        self.sender = sender
        self.address_books = address_books
        pass

    def pred(self, X):
        res = []#np.empty((X.shape[0],10))
        for x in X.values:
            res_temp=[]
            sender=x[3]
            name_list=extract_names(x[2]) #extract name from body
            score={}
            for r in address_books[sender]:
                rec=r[0]
                score[rec]=0
                for name in name_list.split(','):
                    if name in recipients_link[rec]:
                        score[rec]=score[rec]+1
            score=sorted(score.items(), key=operator.itemgetter(1), reverse = True)
            count=0
            if len(score)>0:
                s=score[count][1]
            else:
                s=0
            while s>0 and count<10:
                res_temp.append(score[count][0])
                count=count+1
                if len(score)>count:
                    s=score[count][1]
                else:
                    s=0
            if len(res_temp) < 10:
                res_temp.extend(complete_prediction(10-len(res_temp),self.sender, self.address_books, res_temp))
            res.append(res_temp)
        return res
