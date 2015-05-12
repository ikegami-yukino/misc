# -*  coding: utf-8 -*-
from .lib_bitarray import BitArray


class TestBitArray(object):

    def test___int__(self):
        ba = BitArray(3)
        ba[1] = True
        assert int(ba) == 2

    def test__add__(self):
        ba = BitArray(3)
        ba[0] = True  # 01
        assert int(ba + 1) == 2
        assert int(ba + ba) == 2

    def test__iadd__(self):
        ba = BitArray(2)
        ba += 1  # 00 + 01
        assert ba[0] == True

        other_ba = BitArray(2)
        other_ba[0] = True
        ba += other_ba  # 1 + 1
        assert ba[0] == False
        assert ba[1] == True

    def test__sub__(self):
        ba = BitArray(3)
        ba[1] = True  # 010
        assert int(ba - 1) == 1
        assert int(ba - ba) == 0

    def test__isub__(self):
        ba = BitArray(2)
        ba[1] = True
        ba -= 1
        assert ba[0] == True

        other_ba = BitArray(2)
        other_ba[0] = True
        ba -= other_ba  # 1 - 1
        assert ba[0] == False

    def test_from_int(self):
        ba = BitArray(2)
        ba = ba.from_int(ba, 3)
        assert [ba[0], ba[1]] == [True]*2
