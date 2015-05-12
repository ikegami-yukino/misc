# -*- coding: utf-8 -*-
from collections import defaultdict
import itertools
import numpy as np

"""
Probabilistic Latent Semantic Visualization (PLSV)

IWATA, Tomoharu; YAMADA, Takeshi; UEDA, Naonori.
Probabilistic latent semantic visualization: topic model for visualizing documents.
In: Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining.
ACM, 2008. pp. 363-371.

岩田具治, 山田武士, 上田修功. トピックモデルに基づく文書群の可視化
情報処理学会論文誌, Vol. 50, No. 6, pp. 1234-1244 (June 2009)
http://www.kecl.ntt.co.jp/as/members/iwata/plsv.pdf
"""


class PLSV(object):

    def __init__(self, corpus, dimension=2, k=3, alpha=0.01,
                 beta=0.0001, gamma=0.0001, iteration=10, learning_rate=0.1):
        """
        Params:
            <list><list> corpus : text dataset
            <int> dimension : dimension for visualization
            <int> k : num of topics
            <float> alpha : hyper parameter of theta
            <float> beta : hyper parameter of phi
            <float> gamma : hyper parameter of xai
            <int> iteration : num of learning
            <float> learning_rate
        """
        self.doc_num = len(corpus)
        self.vocas = self.get_vocabularies(corpus)
        self.wordsize = len(self.vocas)

        self.dimension = dimension
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma * k
        self.iteration = iteration
        self.learning_rate = learning_rate

        self.prob_zpx = np.zeros([self.doc_num, self.k])
        self.prob_zpnm = np.random.dirichlet(np.ones(self.k), (self.doc_num, self.wordsize))

        self.phi = np.ones([self.k, self.dimension])
        self.xai = np.zeros([self.doc_num, self.dimension])
        self.theta = np.random.dirichlet(np.ones(self.wordsize), self.k)

    def get_vocabularies(self, corpus):
        vocas = defaultdict(lambda: len(vocas))
        for doc in corpus:
            for word in doc:
                vocas[word]
        return vocas

    def dist(self, doc_id, topic_id):

        def euclid(a, b):
            return np.linalg.norm(a - b)

        denominator = 0
        for t in range(self.k):
            denominator += np.exp(-0.5 * euclid(self.xai[doc_id], self.phi[t]))
        numerator = np.exp(-0.5 * euclid(self.xai[doc_id], self.phi[topic_id]))
        return numerator / denominator

    def posterior(self, d_id, topic_id, word_id):
        denominator = 0
        for t_id in range(self.k):
            denominator += self.prob_zpx[d_id][t_id] * self.theta[t_id][word_id]
        numerator = self.prob_zpx[d_id][topic_id] * self.theta[topic_id][word_id]
        return numerator / denominator

    def expectation(self, corpus):
        for (d_id, t_id) in itertools.product(range(self.doc_num), range(self.k)):
            self.prob_zpx[d_id][t_id] = self.dist(d_id, t_id)

        for (d_id, doc) in enumerate(corpus):
            for (word, t_id) in itertools.product(doc, range(self.k)):
                w_id = self.vocas[word]
                self.prob_zpnm[d_id][w_id][t_id] = self.posterior(d_id, t_id, w_id)

    def theta_update(self, corpus, topic_id, word_id):
        numerator = 0
        denominator = 0
        for (doc_id, doc) in enumerate(corpus):
            for w_id in [self.vocas[word] for word in doc]:
                if w_id == word_id:
                    numerator = self.prob_zpnm[doc_id][word_id][topic_id]
                denominator += self.prob_zpnm[doc_id][word_id][topic_id]
        return (numerator + self.alpha) / (denominator + self.alpha * self.wordsize)

    def xai_update(self, doc_id, topic_id, grad):
        for i in range(self.dimension):
            diff = grad * (self.xai[doc_id][i] - self.phi[topic_id][i]) - self.gamma * self.xai[doc_id][i]
            self.xai[doc_id][i] += self.learning_rate * diff

    def phi_update(self, doc_id, topic_id, grad):
        for i in range(self.dimension):
            diff = grad * (self.phi[topic_id][i] - self.xai[doc_id][i]) - self.beta * self.phi[topic_id][i]
            self.phi[topic_id][i] += self.learning_rate * diff

    def update(self, corpus):
        for (doc_id, doc) in enumerate(corpus):
            for (word, topic_id) in itertools.product(doc, range(self.k)):
                word_id = self.vocas[word]
                p_zpx = self.prob_zpx[doc_id][topic_id]
                p_z = self.prob_zpnm[doc_id][word_id][topic_id]
                grad = p_zpx - p_z
                self.xai_update(doc_id, topic_id, grad)
                self.phi_update(doc_id, topic_id, grad)

    def maximization(self, corpus):
        for (t_id, w_id) in itertools.product(range(self.k), range(self.wordsize)):
            self.theta[t_id][w_id] = self.theta_update(corpus, t_id, w_id)
        self.update(corpus)

    def learning(self, corpus):
        for i in range(self.iteration):
            self.expectation(corpus)
            self.maximization(corpus)

    def dump(self):
        print('phi')
        print(self.phi)
        print('theta')
        print(self.theta)
        print('xai')
        print(self.xai)
        print('prob_zpx')
        print(self.prob_zpx)
        print('prob_zpnm')
        print(self.prob_zpnm)


if __name__ == '__main__':
    DATA = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 2, 3]
    ]
    plsv = PLSV(corpus=DATA)
    plsv.learning(DATA)
    plsv.dump()
