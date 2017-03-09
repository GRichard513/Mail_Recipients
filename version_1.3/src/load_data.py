import pandas as pd
import numpy as np
from collections import Counter
import operator

def load_data(path_to_data):
    # load data files, assume utf-8 encoding
    training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0, encoding='utf-8')
    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0, encoding='utf-8')
    test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0, encoding='utf-8')
    test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0, encoding='utf-8')

    # correcting error on training dataset
    training_info.date.replace(to_replace='0001', value='2001', inplace=True, regex=True)
    training_info.date.replace(to_replace='0002', value='2002', inplace=True, regex=True)

    # transforming recipients into a list
    training_info.recipients = training_info.recipients.str.split()

    # exctract recipients data
    y = pd.DataFrame(training_info, columns=['recipients', 'mid'])
    training_info.drop('recipients', axis=1, inplace=True)

    # join train and test files
    X_df = join_data(training_info, training)
    X_sub_df = join_data(test_info, test)

    # remove non authorise adress from y_df (misssing @)
    y_df = clean(y)

    return X_df, X_sub_df, y_df

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

def clean(y_df):
    y = y_df.recipients.values
    for i in range(y.shape[0]):
        y[i] = [recpt for recpt in y[i] if '@' in recpt]
    y_df.recipients = y
    return y_df

def create_address_books(training, y_df):
    # convert training set to dictionary
    # each sender is described with a key and is associated a list of message IDs (mids)
    emails_ids_per_sender = {}

    for index, series in training.iterrows():
        row = series.tolist()
        sender = row[3]
        ids = row[0]
        if sender in emails_ids_per_sender:
            temp = emails_ids_per_sender[sender]
            emails_ids_per_sender[sender] = temp+[ids]
        else:
            emails_ids_per_sender[sender] = [ids]

    # save all unique sender names
    all_senders = emails_ids_per_sender.keys()

    # create address book with frequency information for each user
    address_books = {}
    i = 0

    for sender, ids in emails_ids_per_sender.items():
        recs_temp = []
        for my_id in ids:
            recipients = y_df[y_df.mid==int(my_id)].recipients.values[0]
            recipients = [rec for rec in recipients if '@' in rec]
            recs_temp.append(recipients)
        # flatten
        recs_temp = [elt for sublist in recs_temp for elt in sublist]
        # compute recipient counts
        rec_occ = dict(Counter(recs_temp))
        # order by frequency
        sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse = True)
        # save
        address_books[sender] = sorted_rec_occ
        i+=1

    return address_books
