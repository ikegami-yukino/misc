# -*- coding: utf-8 -*-
import MeCab
import mmh3
import numpy as np

t = MeCab.Tagger()
t.parse('')


def sentence2array(seentence, n_features):
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
        words.append(mmh3.hash(word) % n_features)
        prev_word = word
    return np.array(words, dtype='int32')
