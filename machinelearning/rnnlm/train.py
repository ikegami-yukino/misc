#!/usr/bin/env python
import argparse
from itertools import chain
import math
import os

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax, Nadam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical


def read_lines(path):
    with open(path, encoding='utf8') as fd:
        for line in fd:
            yield line.rstrip()


def convert(data_path, delimiter, batch_size, tokenizer, max_length, save_path):
    sequences = None
    words = []
    for (i, line) in enumerate(read_lines(data_path), start=1):
        words.append(line.split(delimiter))
        if i % batch_size == 0:
            encoded = tokenizer.texts_to_sequences(words)
            if sequences is None:
                sequences = pad_sequences(encoded, maxlen=max_length, padding='pre')
            else:
                s = pad_sequences(encoded, maxlen=max_length, padding='pre')
                sequences = np.vstack((sequences, s))
            words = []
    encoded = tokenizer.texts_to_sequences(words)
    s = pad_sequences(encoded, maxlen=max_length, padding='pre')
    if sequences is None:
        sequences = s
    else:
        sequences = np.vstack((sequences, s))
    np.savez_compressed(save_path, x=sequences)
    return sequences


class DataGenerator(Sequence):
    def __init__(self, data, batch_size, max_length, vocab_size, tokenizer):
        self.data = data
        self.batch_size = batch_size
        self.data = data
        self.data_length = self.data.shape[0]
        self.offsets = [i * self.data_length // batch_size for i in range(batch_size)]
        self.length = int(np.ceil(self.data_length / self.batch_size))
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.iteration = 0

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        sequences = np.array([self.data[(offset + self.iteration) % self.data_length]
                             for offset in self.offsets])
        self.iteration += 1

        # split into input and output elements
        X, y = sequences[:, :-1], sequences[:, 1:]
        y = to_categorical(y, num_classes=self.vocab_size)
        return X, y


def tokenize(train_path, dev_path, delimiter):
    tokenizer = Tokenizer(split=delimiter, oov_token='<UNK>')
    tokenizer.fit_on_texts(chain(read_lines(train_path), read_lines(dev_path)))
    return tokenizer


def select_optimizer(optimizer):
    if optimizer == 'adam':
        return Adam
    elif optimizer == 'adamax':
        return Adamax
    elif optimizer == 'nadam':
        return Nadam
    raise ValueError


def perplexity(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    ppl = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNNLM')
    parser.add_argument('--train_path', type=str,
                        default='train.tsv', help='Path to the train file')
    parser.add_argument('--dev_path', type=str,
                        default='dev.tsv', help='Path to the dev set file')
    parser.add_argument('--test_path', type=str,
                        default='dev.tsv', help='Path to the test set file')
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
    parser.add_argument('--optimizer', type=str, default='nadam',
                        help='Optimizer function [adam, adamax, nadam]')
    parser.add_argument('--clipnorm', type=float, default=5.0,
                        help='A maximum norm')
    parser.add_argument('--delimiter', type=str, default='\t',
                        help='delimiter in input file')
    args = parser.parse_args()

    tokenizer = tokenize(args.train_path, args.dev_path, args.delimiter)
    open(os.path.join(args.out, 'tokenizer.json'), 'w').write(tokenizer.to_json())
    vocab_size = len(tokenizer.word_index) + 1

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, args.embedding,
                        input_length=args.length - 1))
    model.add(LSTM(args.unit, input_shape=(args.length - 1, args.embedding),
                   dropout=args.dropout, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))

    optimizer = select_optimizer(args.optimizer.lower())
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer(clipnorm=args.clipnorm),
                  metrics=[perplexity])
    model.summary()

    # load/preprocessing
    datum = {}
    for data in ('train', 'dev', 'test'):
        path = os.path.join(args.out, data)
        if os.path.exists(path + '.npz'):
            datum[data] = np.load(path + '.npz')['x']
        else:
            data_path = [x for x in (args.train_path, args.dev_path, args.test_path) if data in x][0]
            datum[data] = convert(data_path, args.delimiter, args.batch, tokenizer, args.length, path)


    # fit network
    train_generator = DataGenerator(datum['train'], args.batch, args.length,
                                    vocab_size, tokenizer)
    valid_generator = DataGenerator(datum['dev'], args.batch, args.length,
                                    vocab_size, tokenizer)

    model.fit_generator(generator=train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=int(np.ceil(datum['train'].shape[0] / args.batch)),
                        validation_steps=int(np.ceil(datum['dev'].shape[0] / args.batch)),
                        epochs=args.epoch,
                        use_multiprocessing=False, verbose=1)

    # Test the model
    test_generator = DataGenerator(datum['test'], args.batch, args.length,
                                   vocab_size, tokenizer)
    results = model.evaluate_generator(generator=test_generator,
                                       steps=int(np.ceil(datum['test'].shape[0] / args.batch)))
    print('loss: %s' % results[0])
    print('perplexity: %s' % results[1])

    # Save the model to files
    open(os.path.join(args.out, 'rnnlm.yaml'), 'w').write(model.to_yaml())
    model.save_weights(os.path.join(args.out, 'rnnlm.hdf5'))
