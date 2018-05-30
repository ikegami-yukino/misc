# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
import numpy as np
import glob
import os
import sys
import MeCab
import mmh3
from sklearn.cross_validation import train_test_split

N_FEATURES = 50000
MAX_LEN = 140
BATCH = 256
EPOCH = 3

t = MeCab.Tagger()
t.parse('')


def sentence2array(seentence):
    words = []
    prev_word = ''
    lines = t.parse(seentence).strip().splitlines()[:-1]
    for l in lines:
        (word, pos) = l.split('\t')
        if word.isdigit():
            word = '<NUM>'
        elif ',固有名詞,人名,' in pos:
            word = '<PERSON>'
            if prev_word == word:
                continue
        words.append(mmh3.hash(word) % N_FEATURES)
        prev_word = word
    return np.array(words, dtype='int32')


def vectorize(path, label):
    X  = []
    y = []
    with open(path) as fd:
        for line in fd:
            line = line.strip()
            X.append(sentence2array(line))
            y.append(label)
    return (X, y)


if __name__ == '__main__':
    (x, y) = vectorize("negative.txt", 0)
    (xx, yy) =  vectorize("positive.txt", 1)
    x += xx
    y += yy
    x = np.array(x)
    y = np.array(y)
    X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)
    model = Sequential()
    model.add(Embedding(N_FEATURES, 256, input_length=MAXLEN))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  class_mode="binary", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
              validation_data=(X_test, y_test))
    model.evaluate(X_test, y_test, batch_size=batch_size)
    open('sentiment_model.yaml', 'w').write(model.to_yaml())
    model.save_weights('sentiment_weights.hdf5')
