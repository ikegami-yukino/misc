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
        def sigmoid(z):
            return 1.0 / (1 + np.exp(-z * self.gain))
        return sigmoid(np.inner(x, w))

    def _update_weight(self, cls, f):
        z = self.z[cls][f]
        if np.abs(z) > self.LAMBDA_ONE:
            self.w[cls][f] -= self.eta[cls][f] * (z - (np.sign(z) * self.LAMBDA_ONE))
        else:
            self.w[cls][f] = 0

    def train(self, x):
        data_order = list(range(len(x)))
        for t in range(self.loop):
            np.random.shuffle(data_order)
            for d in data_order:
                feature = x[d][1:]
                t = x[d][0]
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


if __name__ == '__main__':
    X = np.array(
        (
            (0, 1, 0, 0, 0, 1),
            (0, 1, 0, 0, 0, 1),
            (1, 0, 1, 0, 0, 1),
            (1, 1, 1, 0, 0, 1),
            (2, 0, 0, 1, 0, 1),
            (2, 0, 1, 1, 1, 1),
        )
    )

    k = 3
    ftrlp = FTRLProximal(k, 5, i=200, alpha=0.9)
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
