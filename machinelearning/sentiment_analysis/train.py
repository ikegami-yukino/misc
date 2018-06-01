# -*- coding: utf-8 -*-
import numpy as np

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split

import util

N_FEATURES = 50000
MAX_LEN = 140
BATCH = 20
EPOCH = 3
EMBEDDING_OUT_DIM = 256
LSTM_UNITS = 128
DROPOUT_RATE = 0.5


def vectorize(path, label):
    X = []
    y = []
    with open(path) as fd:
        for line in fd:
            line = line.strip()
            X.append(util.sentence2array(line, N_FEATURES))
            y.append(label)
    return (X, y)


if __name__ == '__main__':
    (x, y) = vectorize("negative.txt", 0)
    (xx, yy) = vectorize("positive.txt", 1)
    x += xx
    y += yy
    x = np.array(x)
    y = np.array(y)
    (X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    model = Sequential()
    model.add(Embedding(N_FEATURES, EMBEDDING_OUT_DIM, input_length=MAX_LEN))
    model.add(LSTM(LSTM_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  class_mode='binary', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=BATCH, nb_epoch=EPOCH,
              validation_data=(X_test, y_test))
    model.evaluate(X_test, y_test, batch_size=BATCH)

    open('sentiment_model.yaml', 'w').write(model.to_yaml())
    model.save_weights('sentiment_weights.hdf5')
