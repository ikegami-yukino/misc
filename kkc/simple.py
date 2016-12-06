# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import pickle
import os
import gzip

DELIMITER = '/'
BT = 'BT'
UT = 'UT'
UTMAXLEN = 4


class SIMPLE(object):

    def __init__(self, simple_file='simple.data'):

        self.simple_file = simple_file
        if os.path.exists(simple_file):
            simple_data = pickle.load(gzip.open(simple_file))
            (self.freq_sigma, self.model, self.kkcdict) = simple_data
        else:
            self.model = defaultdict(int)
            self.kkcdict = defaultdict(list)

        self.CharLogP = math.log(1+sum(self.KKCInput()))

    def KKCInput(self):
        LATINU = list(u'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ')
        NUMBER = list(u'0123456789')
        HIRAGANA = list(u'ぁぃぅぇぉあいうえおかきくけこさしすせそたちつってと'
                        u'なにぬねのはひふへほまみむめもやゆよゃゅょらりるれろ'
                        u'わゎゐゑをんがぎぐげござじずぜぞだぢづでどばびぶべぼ'
                        u'ぱぴぷぺぽ')
        OTHERS = list(u'ヴヵヶー＝￥｀「」；’、。！＠＃＄％＾＆＊（）＿＋｜〜'
                      u'｛｝：”＜＞？・')
        return (len(LATINU), len(NUMBER), len(HIRAGANA), len(OTHERS))

    def update_model(self, corpus, encoding='eucjp'):
        if not self.model:
            self.model = defaultdict(int)
        for line in corpus:
            pairs = line.decode(encoding).strip().split()
            for pair in pairs:
                self.model[pair] += 1
            self.model[BT] += 1
        self.smoothing()
        self.gen_dict()
        self.save()

    def smoothing(self):
        self.freq_sigma = 0  # f() = Σf(word/kkci)
        for pair in self.model.keys():
            freq = self.model[pair]
            self.freq_sigma += freq
            if freq == 1:               # 頻度が１の場合
                self.model[UT] += freq  # f(UT) に加算して
                del self.model[pair]    # f(pair) を消去

    def gen_dict(self):
        for pair in self.model.keys():
            if pair in (BT, UT):
                continue
            kkci = pair.split(DELIMITER)[1]
            self.kkcdict[kkci].append(pair)

    def save(self):
        pickle.dump((self.freq_sigma, self.model, self.kkcdict),
                    gzip.open(self.simple_file, 'w'))

    def add_word(self, word_yomi_pair, frequency=2):
        self.model[word_yomi_pair] += frequency
        self.freq_sigma += frequency
        kkci = word_yomi_pair.split(DELIMITER)[1]
        if word_yomi_pair not in self.kkcdict[kkci]:
            self.kkcdict[kkci].append(word_yomi_pair)
        self.save()

    def conv(self, sent):
        POSI = len(sent)                             # 解析位置 $posi の最大値

        VTable = [list() for i in xrange(POSI + 1)]  # Viterbi Table
        VTable[0].append((None, BT, 0))  # DP左端

        for posi in xrange(1, POSI+1):               # 解析位置(辞書引き右端)
            for from_ in xrange(posi):               # 開始位置(辞書引き左端)
                kkci = sent[from_: from_+(posi-from_)]
                for pair in self.kkcdict[kkci]:      # 既知語のループ
                    best = (None, None, 0)
                    for node in VTable[from_]:
                        logP = self.calc_logp(node[2], self.model[pair],
                                              self.freq_sigma)
                        if (not best[1]) or (logP < best[2]):
                            best = (node, pair, logP)
                    if best[1]:  # 最良のノードがある場合
                        VTable[posi].append(best)

                if (posi - from_) <= UTMAXLEN:  # 未知語によるノード生成
                    best = (None, None, 0)  # 最良のノード(の初期値)
                    for node in VTable[from_]:
                        other = ((posi - from_ + 1) * self.CharLogP)
                        # 入力記号と単語末のBT の生成
                        logP = self.calc_logp(node[2], self.model[UT],
                                              self.freq_sigma, other)
                        if (not best[1]) or (logP < best[2]):
                            pair = '/'.join((kkci, UT))
                            best = (node, pair, logP)
                    if best[1]:         # 最良のノードがある場合
                        VTable[posi].append(best)

        best = (None, None, 0)                      # 最良のノード(の初期値)
        for node in VTable[posi]:
            logP = self.calc_logp(node[2], self.model[BT], self.freq_sigma)
            if (not best[1]) or (logP < best[2]):     # $BT への遷移
                best = (node, BT, logP)

        result = []
        node = best[0]
        while node[0]:
            # result.insert(0, node[1].split('/')[0])
            result.append(node[1])
            node = node[0]
        return ' '.join(result[::-1])

    @staticmethod
    def calc_logp(node_logp, numerator, denominator, other=0):
        return node_logp - math.log(float(numerator) / denominator) + other

if __name__ == '__main__':
    import fileinput
    s = SIMPLE()
    s.update_model(fileinput.input())
    s.add_word(u'もも/もも', 3)
    print len(s.model)
    print s.conv(u'すもももももももものうち')
