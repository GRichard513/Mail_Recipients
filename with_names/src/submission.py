import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import sys

import feature_extractor
import predictor
import time
import load_data

path_to_data = '../data/'

def join_data(X1_df, X2_df):
    # join file and info file. Return a pandas DataFrame with additional features for sender
    df_ar = np.empty((X1_df.shape[0],2), dtype=object)
    X2_arr = X2_df.values
    i = 0
    k = 0
    while i < X1_df.shape[0]:
        for mid in X2_arr[k,1].split():
            df_ar[i] = [X2_arr[k,0],int(mid)]
            i+=1
        k+=1
    df = pd.DataFrame(df_ar, columns=['sender','mid'])
    X1_df = X1_df.merge(df, on='mid')
    return X1_df

def numList(num):
    dic = {}
    for send in num:
        if send in dic.keys():
            dic[send] += 1
        else:
            key = send
            value = 1
            dic[key] = value
    return dic


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

if __name__ == '__main__':
    # load files
    training, training_info, test, test_info, y_df = load_data(path_to_data)

    # join train and test files
    X_df = join_data(training_info, training)
    X_sub_df = join_data(test_info, test)

    # splitting data for cross validation
    skf = ShuffleSplit(n_splits=2, test_size=0.2)
    for train_is, test_is in skf.split(y_df):
        print('--------------------------')
        X_train_df = X_df.iloc[train_is].copy()
        y_train_df = y_df.iloc[train_is].copy()
        X_test_df = X_df.iloc[test_is].copy()
        y_test_df = y_df.iloc[test_is].copy()

        # exctracting features
        print('exctracting features ...')
        fe_train = feature_extractor.FeatureExtractor()
        fe_train.fit(X_train_df, y_train_df)
        X_train_array = fe_train.transform(X_train_df)

        fe_test = feature_extractor.FeatureExtractor()
        fe_test.fit(X_test_df, y_test_df)
        X_test_array = fe_test.transform(X_test_df)

        print(X_test_array.shape)

        print("training and testing ...")
        pdt = {}
        accuracy = {}
        accuracy_TOT = 0
        sender_test = numList(X_test_df.sender.values)
        y_pred = np.zeros(X_test_df.shape[0])

        for sender in sender_test.keys():
            print(sender)
            # the indices corresponding to the sender
            sender_id = np.array(X_test_df.values[:,3] == sender)
            print(sender_id.shape)
            pdt[sender] = predictor.Predictor(X_train_array, y_train_df[sender_id], sender)
            #pdt[sender].fit(X_train_array, y_train_df)
            y_pred[sender_id] = pdt[sender].predict(X_test_array[sender_id])
            accuracy[sender] = mapk(y_test[is_sender], y_pred[is_sender])
            accuracy_TOT += error[sender]
            print('error %s= %.1f%%' %(100. * accuracy, sender))

        print('--------------------------')
        print('error TOT = %.1f%%' %(100. * accuracy_TOT))
