#!/usr/bin/env python
import argparse
import os

import MeCab

import joblib
from keras.models import model_from_yaml
from keras.preprocessing.sequence import pad_sequences


def load_model(model_dir):
    model = model_from_yaml(open(os.path.join(model_dir, 'rnnlm.yaml')).read())
    model.load_weights(os.path.join(model_dir, 'rnnlm.hdf5'))
    return model


def load_tokenizer(model_dir):
    tokenizer = joblib.load(os.path.join(model_dir, 'tokenizer.pkl'))
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return tokenizer, index_to_word


def predict(tagger, words, tokenizer, length, model, index_to_word):
    # tokenize
    words = tagger.parse(words).rstrip().split('\t')
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences(words)
    # pre-pad sequences to a fixed length
    encoded = pad_sequences(encoded, maxlen=length-1, padding='pre')
    # predict probabilities for each word
    proba = model.predict_proba(encoded)
    return [index_to_word.get(idx, '<UNK>') for idx in proba[0].argsort()[-10:][::-1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the next word by RNNLM')
    parser.add_argument('--model_dir', type=str, default='result',
                        help='Directory to output contining the model')
    parser.add_argument('--mecab_args', type=str, default='',
                        help='Arguments of MeCab tokenizer')
    parser.add_argument('--length', type=int, default=20,
                        help='Number of sentence length')
    args = parser.parse_args()

    model = load_model(args.model_dir)
    tokenizer, index_to_word = load_tokenizer(args.model_dir)
    tagger = MeCab.Tagger('-F "%m\t" --eos-format="\n" ' + args.mecab_args)
    tagger.parse('')  # For avoiding bug

    while True:
        words = input('> ')
        print(predict(tagger, words, tokenizer, args.length, model, index_to_word))
