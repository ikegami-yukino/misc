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

    def __init__(self, k, n, i=10, alpha=0.01, beta=1.0, l1=1.0, l2=1.0):
        self.k = k  # num of classes
        self.n = n  # num of features
        self.loop = i
        self.l1 = l1
        self.l2 = l2
        self.a = alpha
        self.b = beta

        self.bias = np.random.rand(k)
        self.w = np.zeros((k, n), dtype=np.float64)
        self.c = np.zeros((k, n), dtype=np.float64)
        self.z = np.zeros((k, n), dtype=np.float64)

    def predict(self, x):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        sign = np.ones_like(self.w)
        sign[np.where(self.w < 0)] = -1
        self.z[np.where(sign * self.w <= self.l1)] = 0
        i = np.where(sign * self.w > self.l1)
        self.z[i] = (sign[i] * self.l1 - self.w[i]) / \
                        ((self.b + np.sqrt(self.c[i])) / self.a + self.l2)
        return softmax(np.dot(self.z, x) + self.bias)

    def train_one(self, feature, y):
        pred_result = self.predict(feature)

        self.bias -= np.mean(pred_result, axis=0)

        t = np.zeros((self.k, self.n))
        t[y] = feature
        e = pred_result[:, np.newaxis] - t
        e2 = e**2
        s = (np.sqrt(self.c + e2) - np.sqrt(self.c)) / self.a
        self.w += e - s * self.z
        self.c += e2

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
    ftrlp = FTRLProximal(k, 4, i=100, alpha=0.01)
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
