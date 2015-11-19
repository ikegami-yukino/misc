"""Morris Counter

Robert Morris. Counting large numbers of events in small registers.
Communications of the ACM, 1978.
"""
from __future__ import unicode_literals
import array
import hashlib
import random


class MorrisCounter(object):
    __REPR_FORMAT__ = '<MorrisCounter; items=%d, radix=%d>'

    def __init__(self, radix=1, n_items=2000000):
        self.radix = radix
        typecode = array.typecodes[1]  # 'B' unsigned char
        self.max_exponent = 255        # maximum of unsigned char
        self.exponents = array.array(typecode, [0]*n_items)
        self.n_items = n_items

    def hashing(self, item):
        item = item.encode()
        return int(hashlib.md5(item).hexdigest(), 16) % self.n_items

    def delta(self, idx):
        return self.radix**-self.exponents[idx]

    def incr(self, item):
        idx = self.hashing(item)
        self.exponents[idx] += (self.exponents[idx] < self.max_exponent and
                                random.random() < self.delta(idx))

    def get(self, item):
        idx = self.hashing(item)
        return self.radix**self.exponents[idx] / (self.radix - 1)

    def __repr__(self):
        return self.__REPR_FORMAT__ % (self.n_items, self.radix)


if __name__ == '__main__':
    mc = MorrisCounter(radix=2)
    [mc.incr('A') for _ in range(2000)]
    [mc.incr('B') for _ in range(2000)]
    print(mc)
    print('Item A: count=%d, exponent=%d' % (mc.get('A'),
                                             mc.exponents[mc.hashing('A')]))
    print('Item B: count=%d, exponent=%d' % (mc.get('B'),
                                             mc.exponents[mc.hashing('B')]))
