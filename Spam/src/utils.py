## utils for reading the emails from various sources
#https://github.com/tuanavu/udacity-course/blob/master/intro_to_machine_learning/lesson/lesson_10_text_learning/vectorize_text.py
#https://gtraskas.github.io/post/spamit/
#https://towardsdatascience.com/k-means-clustering-8e1e64c1561c
import os
import pickle
from itertools import chain
import random
import nltk
import numpy as np
from nltk import tokenize

ENRON_PATH = 'data/Enron/'

def load_enron_vectorized(enron_ids, n, shared=False):


    enrons = [load_text(enron_id) for enron_id in enron_ids]
    flattened_enron = chain.from_iterable(chain.from_iterable(enrons))
    mfw = most_freq_words(flattened_enron, n)

    count = sum((len(h)+len(s) for h,s in enrons)) #number of emails

    X = np.ndarray([count, n])
    Y = np.ndarray([count, 2])

    i = 0
    for nom,anom in enrons:
        X[i:i+len(nom),:] = vectorize_text(nom, mfw)
        Y[i:i+len(nom),:] = vectorize_class(nom, is_anom=False)
        i += len(nom)

        X[i:i+len(anom),:] = vectorize_text(anom, mfw)
        Y[i:i+len(anom),:] = vectorize_class(anom, is_anom=True)
        i += len(anom)



    return X,Y


def vectorize_text(all_text, most_freq_words):

    n = len(all_text)
    m = len(most_freq_words)

    V = np.zeros([n, m], dtype='float32')
    for i,text in enumerate(all_text):
        words = tokenize.word_tokenize(text)
        l_words = {w.lower() for w in words}

        for j,mfw in enumerate(most_freq_words):
            if mfw in l_words:
                V[i][j] = 1.0

    return V


def vectorize_class(all_text, is_anom):
 
    n = len(all_text)
    V = np.zeros([n, 2])

    if is_anom:
        V[:,1]=1.0
    else:
        V[:,0]=1.0
    return V

def load_text(enron_id):

    if not (isinstance(enron_id, int) and 1<=enron_id<=6):
        raise ValueError('enron_id must be an int from 1 to 6') 

    nom = []
    anom = []

    nom_path = ENRON_PATH+'enron{}/ham/'.format(enron_id)
    try:
        for filepath in os.listdir(nom_path):
            try:
                with open(nom_path+filepath, 'r') as fd:
                    txt = fd.read()
                    nom.append(txt)
            except UnicodeDecodeError:
                print('utf-8 error. Skipping file {}'.format(nom_path+filepath))
    except FileNotFoundError:
        print('No enron nom dataset found for id {}'.format(enron_id))
        raise

    anom_path = ENRON_PATH+'enron{}/spam/'.format(enron_id)
    try:
        for filepath in os.listdir(anom_path):
            try:
                with open(anom_path+filepath, 'r') as fd:
                    txt = fd.read()
                    anom.append(txt)
            except UnicodeDecodeError:
                print('utf-8 error. Skipping file {}'.format(anom_path+filepath))
    except FileNotFoundError:
        print('No enron anom dataset found for id {}'.format(enron_id))
        raise

    return (nom, anom)

def most_freq_words(all_text, n):

    freq_dist = nltk.FreqDist()

    for i,text in enumerate(all_text):
        words = tokenize.word_tokenize(text)

        l_words = [w.lower() for w in words
                   if w.isalnum() and not w.isnumeric()]

        freq_dist.update(l_words)

    top_n = [w[0] for w in freq_dist.most_common(n)]
    top_n.sort()
    return top_n