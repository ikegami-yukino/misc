# -*- coding: utf-8 -*-
import numpy as np


class FTRLProximal:
    """
    Multi class FTRLProximal

    This code supports the algorithm of the following paper to multi class prediction:
    McMahan, H. B. et al. (2013, August).
    Ad click prediction: a view from the trenches.
    In Proceedings of the 19th ACM SIGKDD (pp. 1222-1230). ACM.
    http://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/41159.pdf
    """

    def __init__(self, k, n, i=10, alpha=0.01, beta=1.0):
        self.k = k  # num of classes
        self.n = n  # num of features
        self.loop = i
        self.l1 = 1.0
        self.alpha = alpha
        self.beta = beta

        self.bias = np.random.rand(k)
        self.w = np.zeros((k, n), dtype=np.float16)
        self.prev_eta = self.eta = np.ones((k, n), dtype=np.float16)
        self.z = np.zeros((k, n), dtype=np.float16)
        self.grad_square_sum = np.zeros((k, n), dtype=np.float16)  # Σ glad**2

    def predict(self, x):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        return softmax(np.dot(self.w, x) + self.bias)

    def _update_weight(self):
        self.w[np.where(self.z <= self.l1)] = 0
        indices = np.where(self.z > self.l1)
        self.w[indices] += self.eta[indices] * (self.z[indices] - (np.sign(self.z[indices]) * self.l1))

    def train_one(self, feature, y):
        t = np.zeros((self.k, self.n))
        t[y] = feature

        pred_result = self.predict(feature)

        self.bias -= np.mean(pred_result, axis=0)

        weight_grad = t - pred_result[:, np.newaxis]
        self.grad_square_sum += weight_grad**2
        self.eta = self.alpha / (self.beta + np.sqrt(self.grad_square_sum))
        learning_rate = (1. / self.eta) - (1. / self.prev_eta)
        self.z += weight_grad - learning_rate * self.w
        self._update_weight()
        self.prev_eta = self.eta

    def fit(self, X, y):
        num_data = len(X)
        for i in range(self.loop):
            for j in np.random.permutation(num_data):
                self.train_one(X[j], y[j])


if __name__ == '__main__':
    X = np.array(
        (
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
        )
    )
    y = np.array((0, 0, 1, 1, 2, 2))

    k = 3
    ftrlp = FTRLProximal(k, 4, i=1000, alpha=0.01)
    ftrlp.fit(X, y)
    print('WEIGHT:')
    print(ftrlp.w)
    print('BIAS:')
    print(ftrlp.bias)
    print('Z:')
    print(ftrlp.z)
    print('PREDICT:')
    print(ftrlp.predict(np.array((1, 0, 0, 0))))
    print(ftrlp.predict(np.array((0, 1, 0, 0))))
    print(ftrlp.predict(np.array((0, 0, 1, 0))))
    print(ftrlp.predict(np.array((0, 0, 0, 1))))
