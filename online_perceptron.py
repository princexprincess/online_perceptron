#!/usr/bin/env python3
import csv
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data():
    with open('reviews_tr.csv','r') as tr:
        csv_tr = csv.reader(tr)
        header = next(csv_tr)
        labels_tr = [row[0] for row in csv_tr]
        for i in range(len(labels_tr)):
            if labels_tr[i] == '0':
                labels_tr[i] = -1
            elif labels_tr[i] == '1':
                labels_tr[i] = 1
        print(len(labels_tr))
    with open('reviews_tr.csv','r') as tr:
        csv_tr = csv.reader(tr)
        header = next(csv_tr)
        documents_tr = [row[1] for row in csv_tr]
        print(len(documents_tr))
    with open('reviews_te.csv','r') as te:
        csv_te = csv.reader(te)
        header = next(csv_te)
        labels_te = [row[0] for row in csv_te]
        for i in range(len(labels_te)):
            if labels_te[i] == '0':
                labels_te[i] = -1
            elif labels_te[i] == '1':
                labels_te[i] = 1
        print(len(labels_te))
    with open('reviews_te.csv','r') as te:
        csv_te = csv.reader(te)
        header = next(csv_te)
        documents_te = [row[1] for row in csv_te]
        print(len(documents_te))

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(documents_tr + documents_te)
    features_tr = features[0:len(documents_tr)]
    features_te = features[len(documents_tr):]
    print('features done')
    vocab = vectorizer.vocabulary_
    print('vocabulary size: ',len(vocab))

    fw_tr = open('features_tr.txt','wb')
    pickle.dump(features_tr,fw_tr,-1)
    fw_te = open('features_te.txt','wb')
    pickle.dump(features_te,fw_te,-1)
    lw_tr = open('labels_tr.txt','wb')
    pickle.dump(labels_tr,lw_tr,-1)
    lw_te = open('labels_te.txt','wb')
    pickle.dump(labels_te,lw_te,-1)
    print('export done')
    vw = open('vocab.txt','wb')
    pickle.dump(vocab,vw,-1)

def preprocess_data_bigram():
    with open('reviews_tr.csv','r') as tr:
        csv_tr = csv.reader(tr)
        header = next(csv_tr)
        labels = [row[0] for row in csv_tr]
        for i in range(len(labels)):
            if labels[i] == '0':
                labels[i] = -1
            elif labels[i] == '1':
                labels[i] = 1
        print(len(labels))
    with open('reviews_tr.csv','r') as tr:
        csv_tr = csv.reader(tr)
        header = next(csv_tr)
        documents = [row[1] for row in csv_tr]
        print(len(documents))

    bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b')
    features = bigram_vectorizer.fit_transform(documents)
    print('features done')
    vocab = bigram_vectorizer.vocabulary_
    print('vocabulary size: ',len(vocab))

    fw = open('features_bi.txt','wb')
    pickle.dump(features,fw,-1)
    lw = open('labels_bi.txt','wb')
    pickle.dump(labels,lw,-1)
    print('export done')
    vw = open('vocab_bi.txt','wb')
    pickle.dump(vocab,vw,-1)

def load_data():
    fr = open('features_tr.txt','rb')
    features = pickle.load(fr)
    lr = open('labels_tr.txt','rb')
    labels = pickle.load(lr)
    vr = open('vocab.txt','rb')
    vocab = pickle.load(vr)
    return features,labels,vocab

def unigram(features,labels,vocab):
    order = np.linspace(0,999999,1000000).astype(int)
    np.random.shuffle(order)
    return 0

if __name__ == '__main__':
    preprocess_data()
    # preprocess_data_bigram()
    features,labels,vocab = load_data()
    print(features[1].toarray())
    print(labels[1:10])
    print(len(vocab))