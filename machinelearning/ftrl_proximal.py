# -*- coding: utf-8 -*-
import numpy as np


class FTRLProximal:
    """
    McMahan, H. B. et al. (2013, August).
    Ad click prediction: a view from the trenches.
    In Proceedings of the 19th ACM SIGKDD (pp. 1222-1230). ACM.
    http://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/41159.pdf
    """

    def __init__(self, k, n, i=10, alpha=0.05, beta=1.0):
        self.k = k  # num of classes
        self.n = n  # num of features
        self.loop = i
        self.LAMBDA_ONE = 0.2
        self.alpha = alpha
        self.beta = beta
        self.gain = 2

        self.w = np.zeros((k, n), dtype=np.float16)
        self.prev_eta = self.eta = np.ones((k, n), dtype=np.float16)
        self.z = np.zeros((k, n), dtype=np.float16)
        self.g_square_sum = np.zeros((k, n), dtype=np.float16)  # Σ glad**2

    def predict(self, x, w):
        def sigmoid(z, gain):
            return 1.0 / (1 + np.exp(-z * gain))
        return sigmoid(np.inner(x, w), self.gain)

    def _update_weight(self, cls, i):
        z = self.z[cls][i]
        if np.abs(z) > self.LAMBDA_ONE:
            self.w[cls][i] -= self.eta[cls][i] * (z - (np.sign(z) * self.LAMBDA_ONE))
        else:
            self.w[cls][i] = 0

    def train_one(self, t, feature):
        for cls in range(self.k):
            pred_result = self.predict(feature, self.w[cls])
            glad = (pred_result - int(t == cls)) * feature
            self.g_square_sum[cls] += glad**2
            for i in range(self.n):
                self.eta[cls][i] = self.alpha / (self.beta + np.sqrt(self.g_square_sum[cls][i]))
                learning_rate = (1.0 / self.eta[cls][i]) - (1.0 / self.prev_eta[cls][i])
                self.z[cls][i] += glad[i] - learning_rate * self.w[cls][i]
                self._update_weight(cls, i)
                self.prev_eta[cls][i] = self.eta[cls][i]

    def train(self, x):
        learning_order = list(range(len(x)))
        for i in range(self.loop):
            np.random.shuffle(learning_order)
            for j in learning_order:
                self.train_one(x[j][0], x[j][1:])


if __name__ == '__main__':
    X = np.array(
        (
            (0, 1, 0, 0, 0, 1),
            (0, 1, 1, 0, 0, 1),
            (1, 0, 1, 0, 0, 1),
            (1, 0, 1, 1, 0, 1),
            (2, 0, 0, 1, 0, 1),
            (2, 0, 0, 1, 1, 1),
            (3, 0, 0, 0, 0, 1),
        )
    )

    k = 4
    ftrlp = FTRLProximal(k, 5, i=1000, alpha=0.9)
    ftrlp.train(X)
    print('WEIGHT:')
    print(ftrlp.w)
    print('Z:')
    print(ftrlp.z)
    print('PREDICT:')
    print([ftrlp.predict(np.array((1, 0, 0, 0, 1)), ftrlp.w[j]) for j in range(k)])
    print([ftrlp.predict(np.array((0, 1, 0, 0, 1)), ftrlp.w[j]) for j in range(k)])
    print([ftrlp.predict(np.array((0, 0, 1, 0, 1)), ftrlp.w[j]) for j in range(k)])
    print([ftrlp.predict(np.array((0, 0, 0, 1, 1)), ftrlp.w[j]) for j in range(k)])
    print([ftrlp.predict(np.array((1, 1, 1, 1, 1)), ftrlp.w[j]) for j in range(k)])
