"""Morris Counter using dbm"""

from __future__ import unicode_literals
import dbm
import random


class MorrisDBM(object):

    def __init__(self, path, radix=2):
        self.path = path
        self.radix = radix
        self.max_exponent = 255  # maximum of unsigned char

    def delta(self, exponent):
        return self.radix**-exponent

    def get_exponent(self, word):
        with dbm.open(self.path, 'c') as db:
            if word in db:
                return int.from_bytes(db[word], 'little')
        return 0

    def incr(self, word):
        exponent = self.get_exponent(word)
        print(exponent, self.delta(exponent))
        if (exponent < self.max_exponent and random.random() < self.delta(exponent)):
            with dbm.open(self.path, 'c') as db:
                exponent += 1
                db[word] = exponent.to_bytes(self.max_exponent, 'little')

    def get(self, word):
        exponent = self.get_exponent(word)
        return self.radix**exponent / (self.radix - 1)
