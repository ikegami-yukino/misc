# -*- coding: utf-8 -*-
from bitarray import bitarray


class BitArray(bitarray):

    def __init__(self, *args, **kwds):
        self.setall(False)

    def __int__(self):
        binary = self.to01()
        if binary:
            return int(binary[::-1], 2)
        return 0

    def __add__(self, num):
        if isinstance(num, BitArray):
            num = int(num)
        _sum = self.__int__() + num
        new_ba = BitArray(self.length())
        return self.from_int(new_ba, _sum)

    def __iadd__(self, num):
        return self.__add__(num)

    def __sub__(self, num):
        if isinstance(num, BitArray):
            num = int(num)
        diff = self.__int__() - num
        new_ba = BitArray(self.length())
        return self.from_int(new_ba, diff)

    def __isub__(self, num):
        return self.__sub__(num)

    def from_int(self, ba, num):
        ba.setall(False)
        for (i, x) in enumerate(bin(num)[2:][::-1]):
            if x == '1':
                ba.__setitem__(i, True)
        return ba
