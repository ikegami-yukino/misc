# -*- coding: utf-8 -*-
import os

import msgpackrpc
from keras.models import model_from_yaml
from keras.preprocessing import sequence

import util

# Keras
N_FEATURES = 50000
MAXLEN = 140

DATA_DIR = '/work/rpc'
modelpath = os.path.join(DATA_DIR, 'sentiment_model.yaml')
with open(modelpath) as fd:
    model = model_from_yaml(fd.read())
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.load_weights(os.path.join(DATA_DIR, 'sentiment_weights.hdf5'))


class SentimentAnalysis(object):

    def preprocessing(self, line):
        line = line.strip()
        X = [util.sentence2array(line, N_FEATURES)]
        return sequence.pad_sequences(X, maxlen=MAXLEN)

    def predict(self, x):
        x = self.preprocessing(x)
        return model.predict(x)[0][0]


if __name__ == "__main__":
    server = msgpackrpc.Server(SentimentAnalysis())
    server.listen(msgpackrpc.Address('0.0.0.0', 18798))
    server.start()
