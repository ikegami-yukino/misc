import msgpackrpc
import mmh3
from keras.models import model_from_yaml
import os
import MeCab
from keras.preprocessing import sequence
import numpy as np

# Keras
N_FEATURES = 50000
MAXLEN = 140
DATA_DIR = '/work/rpc'
modelpath = os.path.join(DATA_DIR, 'sentiment_model.yaml')
with open(modelpath) as fd:
    model = model_from_yaml(fd.read())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights(os.path.join(DATA_DIR, 'sentiment_weights.hdf5'))

# MeCab
t = MeCab.Tagger()
t.parse('')


class SentimentAnalysis(object):
    def sentence2array(self, sentence):
        words = []
        prev_word = ''
        lines = t.parse(sentence.decode('utf8')).strip().splitlines()[:-1]
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

    def preprocessing(self, line):
        line = line.strip()
        X = [self.sentence2array(line)]
        return sequence.pad_sequences(X, maxlen=MAXLEN)

    def predict(self, x):
        x = self.preprocessing(x)
        return model.predict(x)[0][0]


if __name__ == "__main__":
    server = msgpackrpc.Server(SentimentAnalysis())
    server.listen(msgpackrpc.Address("0.0.0.0", 18798))
    server.start()
