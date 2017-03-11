import string
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

cachedStopWords = stopwords.words("english")

names=pd.read_csv('firstnames.txt',sep="\t")
dict_names=[x.split()[0] for x in names.Name]
dict_names=[x.lower() for x in dict_names]
dict_months=['january','february','march','april','may','june','july','august','september','october','november','december']

#Name extraction from dictionary
def extract_names(text, dict_n=dict_names, dict_m=dict_months):
    name_list=[]
    text=re.sub(r'[^\w\s]',' ',text)
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    
    for z in text.split():
        if (z.lower() in dict_n and z.lower()!=z and z.lower() not in dict_m):
            name_list.append(z)
    return ','.join([word for word in np.unique(name_list).tolist()])

#Name extractor with nltk: does not work well and is very long
def extract_entities(text):
    output_list=[]
    
    s=string.punctuation.replace('@','')
    s=s.replace('+','')
    text= re.sub(r'[^\w\s]',' ',text)
    text = text.translate(str.maketrans('','',s))
    text = " ".join(text.split())
    #text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    #print(text.split()[:10])
    count=0
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            count=count+1
            if count<10:
                if type(chunk)!=tuple:
                    #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                    output_list.append(' '.join(c[0] for c in chunk.leaves()))
    return output_list


def add_proper_names(X_df,inplace=False, extractor='name'):
    print('------------')
    print('Load proper names \n')
    if inplace==True:
        X_tmp=X_df
    else:
        X_tmp=X_df.copy()
    
    X_tmp['proper_names']=['' for i in range(len(X_tmp))]
    
    N=len(X_tmp)
    print('Total length: ', N)
    for i in range(len(X_tmp)):
        #x=X_tmp.loc[i]
        if extractor=='name':
            X_tmp.proper_names[i]=extract_names(X_tmp.body[i])
        if extractor=='entities':
            X_tmp.proper_names[i]=extract_entities(X_tmp.body[i])
        if int(float(i*10/N))>int(float((i-1)*10/N)):
            print(int(float(i*100)/N),'%')
    print('------------')
    return X_tmp

def create_name_dict(X_tmp, y_df):
    X_tmp_list=X_tmp.copy()
    X_tmp_list.proper_names=X_tmp.proper_names.str.split(',')
    X_prop_links=X_tmp_list.merge(y_df, on='mid').drop(['body','date'],axis=1)

    X_prop_links.head()

    surname_link={}

    for x in X_prop_links.values:
        if len(x[3])>0:
            for address in x[4]:
                combined=x[2]+','+address
                if combined not in surname_link:
                    surname_link[combined]=[]
                for name in x[3]:
                    if len(name)>0:
                        surname_link[combined].append(name)

    for combined in surname_link:
        surname_link[combined]=np.unique(surname_link[combined]).tolist()

    recipients_link={}

    for x in X_prop_links.values:
        if len(x[3])>0:
            for rec in x[4]:
                if rec not in recipients_link:
                    recipients_link[rec]=[]
                for name in x[3]:
                    if len(name)>0:
                        recipients_link[rec].append(name)
    for rec in recipients_link:
        recipients_link[rec]=np.unique(recipients_link[rec]).tolist()

    return surname_link, recipients_link
