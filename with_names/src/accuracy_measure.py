import numpy as np

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def customed_map(y_true,y_pred,k=None):
    if k==None:
        k=len(y_pred)
    if len(y_true)!=len(np.unique(y_true)) or len(y_pred)!=len(np.unique(y_pred)):
        print('Error: duplicate values')
        return
    count=0.
    right=0.
    res=0.
    for i in y_pred:
        count=count+1
        if len(np.where(y_true==i)[0])>0:
            right=right+1
            res=res+right/count
    return res/k
