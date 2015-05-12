# -*  coding: utf-8 -*-
import mmh3
from .bloomfilter import BloomFilter, CountingBloomFilter
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


class TestBloomFilter(object):

    def test_init_bitarrays(self):
        bf = BloomFilter()
        bitarrays = bf.init_bitarrays(width=100, num_hashes=2)
        assert list(map(len, bitarrays)) == [100, 100]

    def test__contains__(self):
        bf = BloomFilter()
        bf.add('hoge')
        assert ('hoge' in bf) is True

    def test__str__(self):
        bf = BloomFilter(size=2, num_hashes=2)
        bf.bitarrays[0][0] = True
        bf.bitarrays[1][0] = True
        assert str(bf) == '0101'

    def test__int__(self):
        bf = BloomFilter(size=2, num_hashes=2)
        bf.bitarrays[0][0] = True  # 0100
        assert int(bf) == 4

    def test__len__(self):
        bf = BloomFilter()
        assert len(bf) == 0

        bf.num_keys = 10
        assert len(bf) == 10

    def test_calc_hash(self):
        bf = BloomFilter(size=10, num_hashes=2)
        assert bf.calc_hash('hoge') == [mmh3.hash('hoge', 0) % 10, mmh3.hash('hoge', 1) % 10]

    def test_add(self):
        bf = BloomFilter(size=10, num_hashes=2)
        assert bf.add('hoge') == True
        assert bf.num_keys == 1
        assert list(map(lambda x: x.count(), bf.bitarrays)) == [1, 1]

    def test_from_int(self):
        bf = BloomFilter(size=2, num_hashes=2)
        bf.from_int(5)  # 0101
        assert bf.bitarrays[0][0] is True
        assert bf.bitarrays[1][0] is True

    def test_compress(self):
        bf = BloomFilter(size=13, num_hashes=2)
        bf.from_int(int('0x11110', 16))
        assert bf.compress() == '1-4,0'

    def test_expand(self):
        bf = BloomFilter(size=13, num_hashes=2)
        assert bf.expand('1-4,0') == '11110'


class TestCountingBloomFilter(object):

    def test_calc_hash(self):
        bf = CountingBloomFilter(size=10, num_hashes=2, bits=2)
        hashes = bf.calc_hash('hoge')
        assert hashes[0] == (mmh3.hash('hoge', 0) % 10) * 2
        assert hashes[1] == (mmh3.hash('hoge', 1) % 10) * 2

    def test__contains__(self):
        bf = CountingBloomFilter(size=10, num_hashes=2, bits=2)
        bf.add('hoge')
        assert ('hoge' in bf) is True

    def test_update(self):
        bf = CountingBloomFilter(size=2, num_hashes=2, bits=2)
        hashes = [0, 0]
        assert bf._update('hoge', 'inc', hashes) is True  # Add hoge
        assert bf._update('hoge', 'inc', hashes) is False  # already added
        assert bf._update('hoge', 'dec', hashes) is True  # Declement hoge
        assert bf._update('hoge', 'dec', hashes) is True
        assert bf._update('hoge', 'dec', hashes) is False  # hoge is None
        assert bf._update('hoge', 'inc', hashes) is True  # Add hoge again

    def test_add(self):
        bf = CountingBloomFilter(size=2, num_hashes=2, bits=2)
        hashes = [0, 0]
        assert bf.add('hoge', hashes) is True  # Add hoge
        assert bf.add('hoge', hashes) is False  # already added

    def test_remove(self):
        bf = CountingBloomFilter(size=2, num_hashes=2, bits=2)
        hashes = [0, 0]
        assert bf.remove('hoge', hashes) is False  # hoge doesn't exist
        assert bf.add('hoge', hashes) is True  # Add hoge
        assert bf.remove('hoge', hashes) is True
