#!/usr/bin/env python
import argparse
import os
import sys

import MeCab
import numpy as np

from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json


class NextWordPredictor(object):
    def __init__(self, model_dir):
        self.model = self._load_model(model_dir)
        self.tokenizer = self._load_tokenizer(model_dir)

    @staticmethod
    def _load_model(model_dir):
        model = model_from_yaml(open(os.path.join(model_dir, 'rnnlm.yaml')).read())
        model.load_weights(os.path.join(model_dir, 'rnnlm.hdf5'))
        return model

    @staticmethod
    def _load_tokenizer(model_dir):
        tokenizer = tokenizer_from_json(open(os.path.join(model_dir, 'tokenizer.json')).read())
        return tokenizer

    def predict(self, line, length=20, mecab_args=''):
        # tokenize
        tagger = MeCab.Tagger("-F %m\\t --eos-format=\n " + mecab_args)
        tagger.parse('')  # For avoiding bug
        words = tagger.parse(line).rstrip().split('\t')
        # encode the text as integer
        encoded = self.tokenizer.texts_to_sequences([' '.join(words)])
        # pre-pad sequences to a fixed length
        encoded = pad_sequences(encoded, maxlen=length-1, padding='pre')
        # predict probabilities for each word
        proba = self.model.predict_proba(encoded)
        return self.tokenizer.sequences_to_texts([(-proba)[0][-1].argsort()[:20]])[0].split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the next word by RNNLM')
    parser.add_argument('--model_dir', type=str, default='result',
                        help='Directory to output contining the model')
    parser.add_argument('--mecab_args', type=str, default='',
                        help='Arguments of MeCab tokenizer')
    parser.add_argument('--length', type=int, default=20,
                        help='Number of sentence length')
    parser.add_argument('--input_file', type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args()

    predictor = NextWordPredictor(args.model_dir)
    for line in args.input_file:
        print(predictor.predict(line, args.length, args.mecab_args))
