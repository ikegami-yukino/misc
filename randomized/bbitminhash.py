# -*- coding: utf-8 -*-
"""b-Bit Minwise Hashing

This module expresses document as k-length vector (each value ranges b-bit)
This is based on the following paper:
  Li, P., & König, C. b-Bit minwise hashing.
  In Proc. of the WWW 2010 (pp. 671-680). ACM. (2010, April).
  http://research.microsoft.com/pubs/120078/wfc0398-lips.pdf
"""
from __future__ import division
import numpy as np
import mmh3


class BbitMinHash(object):

    def __init__(self, b=1, k=128, d=1000000):
        """
        Params:
            <int> b : num of bits
            <int> k : num of minhash length
            <int> d : num of vocabularies
        """
        self.b = b
        self.seeds = range(k)
        self.num_hashes = k
        self.d = d
        self.dtype = self._determine_dtype(b)
        self.doc_to_hash = np.vectorize(lambda word, seed: mmh3.hash(word, seed))

    def _determine_dtype(self, b):
        """Determine datatype representing b-Bit MinHash
        Param:
            <int> b
        Return:
            <numpy.dtype> dtype
        """
        if 64 < b or 1 > b:
            raise ValueError('b-Bit size must be 1 <= b <= 64')
        elif b > 32:
            return np.int64
        elif b > 16:
            return np.int32
        elif b > 8:
            return np.int16
        elif b > 1:
            return np.int8
        return np.bool

    def gen_minhash(self, doc):
        """Generating minhash bits
        Param:
            <list> doc
        Return:
            <numpy.array> bits
        """
        bits = np.zeros(self.num_hashes, dtype=self.dtype)
        for (i, seed) in enumerate(self.seeds):
            hashes = self.doc_to_hash(doc, seed)
            min_hash = np.min(hashes)
            b_bit_min_hash = bin(min_hash)[-self.b:]
            if b_bit_min_hash[0] == 'b':
                b_bit_min_hash = b_bit_min_hash[1:]
            bits[i] = int(b_bit_min_hash, 2)
        return bits

    def compute_similarity(self, lhs_minhash, rhs_minhash):
        """Compute similarity between both docs
        Params:
            <numpy.array> lhs_minhash
            <numpy.array> rhs_minhash
        Return:
            <float> similarity
        """
        def compute_A(r):
            numerator = r * (1 - r)**((1 << self.b) - 1)
            denominator = 1 - (1 - r)**(1 << self.b)
            return numerator / denominator

        lhs_r = len(lhs_minhash) / self.d
        rhs_r = len(rhs_minhash) / self.d

        both_r = lhs_r + rhs_r
        lhs_r_ratio = lhs_r / both_r
        rhs_r_ratio = rhs_r / both_r

        lhs_A = compute_A(lhs_r)
        rhs_A = compute_A(rhs_r)

        C1 = lhs_A * rhs_r_ratio + rhs_A * lhs_r_ratio
        C2 = lhs_A * lhs_r_ratio + rhs_A * rhs_r_ratio

        xor = np.bitwise_xor(lhs_minhash, rhs_minhash)
        num_nonzero = np.count_nonzero(xor)
        E = (self.num_hashes - num_nonzero) / self.num_hashes
        similarity = (E - C1) / (1 - C2)
        return similarity if similarity > 0 else 0
