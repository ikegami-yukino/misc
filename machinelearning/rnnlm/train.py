#!/usr/bin/env python
import argparse
import multiprocessing
import math
import os

import numpy as np

import joblib
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam, Adamax, Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, data_path, batch_size, max_length, vocab_size):
        self.data_path = data_path
        self.batch_size = batch_size
        with open(self.data_path) as fd:
            self.data = fd.read().splitlines()
        self.length = int(np.ceil(len(self.data) / self.batch_size))
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        flatten = lambda x: [z for y in x for z in ((y,) if isinstance(y, (str, int)) else flatten(y,))]

        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        sequences = []
        encoded = tokenizer.texts_to_sequences(batch_x)
        sequence = [encoded[:i + 1] for i in range(1, len(encoded))]
        sequence = flatten(sequence)
        sequences.append(sequence)

        sequences = pad_sequences(sequences, maxlen=self.max_length, padding='pre')

        # split into input and output elements
        sequences = np.array(sequences)

        X, y = sequences[:, :-1], sequences[:, -1]
        y = to_categorical(y, num_classes=self.vocab_size)
        return X, y


def tokenaize(train_path, dev_path):
    with open(train_path) as fd:
        data = fd.read()
    with open(dev_path) as fd:
        data += fd.read()
    tokenizer = Tokenizer(split='\t', oov_token='<UNK>')
    tokenizer.fit_on_texts([data])
    return tokenizer


def select_optimizer(optimizer):
    if optimizer == 'adam':
        return Adam
    elif optimizer == 'adamax':
        return Adamax
    elif optimizer == 'nadam':
        return Nadam
    raise ValueError


def calc_data_size(train_path):
    return sum([1 for f in open(train_path).read()])


def perplexity(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    ppl = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNNLM')
    parser.add_argument('--train_path', type=str,
                        default='train.tsv', help='Path to the train tsv file')
    parser.add_argument('--dev_path', type=str,
                        default='dev.tsv', help='Path to the dev set tsv file')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of units')
    parser.add_argument('--embedding', type=int, default=650,
                        help='Number of word embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--length', type=int, default=20,
                        help='Number of sentence length')
    parser.add_argument('--batch', '-b', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        default='nadam',
                        help='Optimizer function [adam, adamax, nadam]')
    parser.add_argument('--clipnorm', type=float, default=5.0,
                        help='A maximum norm')
    args = parser.parse_args()

    tokenizer = tokenaize(args.train_path, args.dev_path)
    joblib.dump(tokenizer, os.path.join(args.out, 'tokenizer.pkl'), compress=True)
    vocab_size = len(tokenizer.word_index) + 1

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, args.embedding,
                        input_length=args.length - 1))
    model.add(LSTM(args.unit, dropout=args.dropout))
    model.add(Dense(vocab_size, activation='softmax'))

    optimizer = select_optimizer(args.optimizer.lower())
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer(clipnorm=args.clipnorm),
                  metrics=[perplexity])

    # fit network
    train_generator = DataGenerator(args.train_path, args.batch, args.length, vocab_size)
    valid_generator = DataGenerator(args.dev_path, args.batch, args.length, vocab_size)

    train_data_size = calc_data_size(args.train_path)
    dev_data_size = calc_data_size(args.dev_path)

    model.fit_generator(generator=train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=int(np.ceil(train_data_size / args.batch)),
                        validation_steps=int(np.ceil(dev_data_size / args.batch)),
                        epochs=args.epoch,
                        use_multiprocessing=False, verbose=1)

    # Save model to file
    open(os.path.join(args.out, 'rnnlm.yaml'), 'w').write(model.to_yaml())
    model.save_weights(os.path.join(args.out, 'rnnlm.hdf5'))
