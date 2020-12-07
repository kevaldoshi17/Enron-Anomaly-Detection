import os
import pickle
from itertools import chain
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import networkx as nx
import random
import nltk
from nltk import tokenize
nltk.download('punkt')
from utils import *
G = nx.Graph()



mfw = None
enron_ids = [1,2,3,4]   
mini_batch_size = 40    
n = 150                 
train_fraction = 0.75   

ENRON_PATH = 'data/Enron/'



os.makedirs(ENRON_PATH, exist_ok=True)





ed = load_enron_vectorized(enron_ids=enron_ids, n=n, shared=False)

print(ed[1][0,:])
N = ed[0].shape[0]


train_fraction = round(N*train_fraction)

training_data = ((ed[0][:train_fraction,:], ed[1][:train_fraction,:]))
test_data = ((ed[0][train_fraction:,:], ed[1][train_fraction:,:]))

print('Using {} messages as training data'.format(train_fraction))





model = Sequential()
model.add(Dense(80, input_dim=150, activation='softmax'))
model.add(Dense(80, activation='softmax'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
epochs, eta = 30, 0.2
   
model.fit(training_data[0], training_data[1], epochs=epochs, batch_size=10)
print()