# -*- coding: utf-8 -*-
import mmh3
from .lib_bitarray import BitArray


class BloomFilter(object):

    def __init__(self, size=5000000, num_hashes=1):
        self.m = size
        self.width = size
        self.num_hashes = num_hashes
        self.bitarrays = self.init_bitarrays(num_hashes, self.width)
        self.num_keys = 0

    def init_bitarrays(self, num_hashes, width):
        return [BitArray(width) for i in range(num_hashes)]

    def __contains__(self, key, hashes=None):
        if not hashes:
            hashes = self.calc_hash(key)
        return all(self.bitarrays[i][x] for (i, x) in enumerate(hashes))

    def __str__(self):
        return ''.join((self.bitarrays[i].to01()[::-1] for i in range(self.num_hashes)))

    def __int__(self):
        return int(self.__str__(), 2)

    def __len__(self):
        return self.num_keys

    def calc_hash(self, key):
        return [mmh3.hash(key, i) % self.m for i in range(self.num_hashes)]

    def add(self, key, hashes=None):
        if not hashes:
            hashes = self.calc_hash(key)
        added = False
        for (i, x) in enumerate(hashes):
            if not self.bitarrays[i][x]:
                self.bitarrays[i][x] = 1
                added = True
        if added:
            self.num_keys += 1
            return True
        return False

    def from_int(self, num):
        self.bitarrays = self.init_bitarrays(self.num_hashes, self.width)
        row = self.num_hashes - 1
        for (i, n) in enumerate(bin(num)[::-1][:-2], start=1):
            if n != '0':
                self.bitarrays[row][(i % self.width) - 1] = True
            if i % self.width == 0:
                row -= 1

    def check_usage(self):
        used = sum(map(lambda x: x.count(), self.bitarrays))
        total = self.width * self.num_hashes
        return float(used) / total

    def compress(self):
        '''Compress bitarrays by Run-length encoding with converting hexadecimal
        '''
        compressed = ''
        count = 1
        prev_char = ''
        hex_num = hex(self.__int__())
        if hex_num[-1] == 'L':
            hex_num = hex_num[:-1]
        for char in hex_num[2:]:
            if char == prev_char:
                count += 1
            else:
                if count > 2:
                    compressed += '%s-%s,' % (prev_char, hex(count)[2:])
                else:
                    compressed += '%s' % (prev_char * count)
                count = 1
            prev_char = char
        if count > 2:
            compressed += '%s-%s' % (char, hex(count)[2:])
        else:
            compressed += '%s' % (char * count)
        return compressed

    def expand(self, compressed):
        def add_repeat(result, repeat_hex):
            repeat = int('0x%s' % repeat_hex, 16) - 1
            if repeat > (1 << 32):
                (quotient, remainder) = divmod(repeat, (1 << 32))
                for i in xrange(quotient):
                    result += result[-1] * (1 << 32)
                repeat = remainder
            return result + result[-1] * repeat

        result = ''
        repeat_hex = ''
        hyphen = False
        for char in compressed:
            if char == '-':
                hyphen = True
            elif char == ',':
                result = add_repeat(result, repeat_hex)
                repeat_hex = ''
                hyphen = False
            elif hyphen:
                repeat_hex += char
            else:
                result += char
        if repeat_hex:
            result = add_repeat(result, repeat_hex)
        return result

    def check_usage(self):
        used = sum(map(lambda x: x.count(), self.bitarrays))
        total = self.width * self.num_hashes
        return float(used) / total


class CountingBloomFilter(BloomFilter):

    def __init__(self, size=3000000, num_hashes=2, bits=3):
        self.m = size
        self.width = size * bits
        self.num_hashes = num_hashes
        self.bits = bits
        self.bitarrays = self.init_bitarrays(num_hashes, self.width)
        self.num_keys = 0
        self.maximum = (0x01 << self.bits) - 1

    def calc_hash(self, key):
        return [(mmh3.hash(key, i) % self.m) * self.bits for i in range(self.num_hashes)]

    def __contains__(self, key, hashes=None):
        if not hashes:
            hashes = self.calc_hash(key)
        return all(any(self.bitarrays[i][x:x + self.bits]) for (i, x) in enumerate(hashes))

    def _update(self, key, oprtr, hashes=None):
        if not hashes:
            hashes = self.calc_hash(key)
        updated = False
        for (i, x) in enumerate(hashes):
            idx_range = slice(x, x + self.bits)
            value = int(self.bitarrays[i][idx_range])
            if oprtr == 'inc':
                if value < self.maximum:
                    if value == 0:
                        updated = True
                    self.bitarrays[i][idx_range] += 1
            elif oprtr == 'dec':
                if value > 0:
                    updated = True
                    self.bitarrays[i][idx_range] -= 1
        return updated

    def add(self, key, hashes=None):
        updated = self._update(key, 'inc', hashes)
        if updated:
            self.num_keys += 1
            return True
        return False

    def remove(self, key, hashes=None):
        updated = self._update(key, 'dec', hashes)
        if updated:
            self.num_keys -= 1
            return True
        return False
