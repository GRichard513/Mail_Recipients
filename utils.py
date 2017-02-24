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
