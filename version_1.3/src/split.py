import pandas as pd
import numpy as np

def split(X_df, n=20):

    X_train = {}
    for sender in X_df.sender.unique().tolist():
        X_train[sender] = X_df[X_df.sender==sender].sample(n=n)

    train_tot = pd.concat([X_train[sender] for sender in X_df.sender.unique().tolist()])
    train_is = []
    test_is = []
    train = X_df.mid.isin(train_tot.mid)

    for i in range(X_df.shape[0]):
        if train.ix[i]:
            test_is.extend([i])
        else:
            train_is.extend([i])

    train_is = np.asarray(train_is)
    test_is = np.asarray(test_is)

    return train_is, test_is
