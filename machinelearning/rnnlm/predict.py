import random
import subprocess

import MeCab
import joblib
from keras.models import model_from_yaml
from keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 40

# RNNLM
model = model_from_yaml(open('rnnlm.yaml').read())
model.load_weights('rnnlm.hdf5')

# Tokenizer
tokenizer = joblib.load('tokenizer.pkl')
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# MeCab
proc = subprocess.Popen('mecab-config --dicdir', shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
(dicdir, _) = proc.communicate()
dicdir = dicdir.decode('utf8').strip()
wakati_tagger = MeCab.Tagger('-O wakati -d %s/mecab-ipadic-neologd' % dicdir)


def predict(sentence):
    # tokenize
    words = wakati_tagger.parse(sentence).rstrip().split()
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences(words)
    # pre-pad sequences to a fixed length
    encoded = pad_sequences(encoded, maxlen=MAX_LENGTH-1, padding='pre')
    # predict probabilities for each word
    proba = model.predict_proba(encoded)
    return [index_to_word[idx] for idx in proba[0].argsort()[-10:][::-1]]


if __name__ == '__main__':
    print(predict('<PLACE>'))
