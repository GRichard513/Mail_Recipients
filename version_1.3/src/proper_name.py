import string
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

cachedStopWords = stopwords.words("english")

names = pd.read_csv('firstnames.txt',sep="\t")
dict_names = [x.split()[0] for x in names.Name]
dict_names = [x.lower() for x in dict_names]
dict_months = ['january','february','march','april','may','june','july','august','september','october','november','december']

#Extract names at the beginning of the document
def extract_names(text, dict_n=dict_names, dict_m=dict_months,nb_words=5):

    name_list=[]
    text=re.sub(r'[^\w\s]',' ',text)
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    forward=False

    if nb_words==None:
        nb_words=len(text.split())

    dear=False

    count=0
    for z in text.split()[:nb_words]:
        if (z.lower()=='forwarded' or z.lower()=='original'):
            forward=True
        if(z.lower()=='dear' or z.lower()=='hi' or z.lower()=='thanks'):
            dear=True

        if (z.lower() in dict_n and z.lower()!=z and z.lower() not in dict_m and forward==False and (dear==True or count==0)):
            name_list.append(z.lower())

    if len(name_list)==0:
        name_list=['']

    return name_list#','.join([word for word in np.unique(name_list).tolist()])

#Add columns to the initial dataframe
def create_names_df(X_df):
    X_names=X_df.copy()

    list_names=[]
    for x in X_names.body:
        l_names=extract_names(x,nb_words=5)
        list_names.append(l_names)
    X_names['names']=list_names
    return X_names

#attributes names for mail addresses
def names(address_books):
    recipient_name = {}
    for sender in address_books:
        for rec, value in address_books[sender]:
            if rec not in recipient_name:
                recipient_name[rec]='DefaultNULL'
                if '.' in rec[:rec.find('@')]:
                    found = rec[:rec.find('.')].lower()
                    if found in dict_names:
                        recipient_name[rec] = found
                    else:
                        found=rec[rec.find('.')+1:rec.find('@')].lower()
                        if found in dict_names:
                            recipient_name[rec] = found
    return recipient_name
