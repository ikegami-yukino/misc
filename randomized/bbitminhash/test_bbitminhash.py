# -*  coding: utf-8 -*-
from nose.tools import assert_equals, assert_almost_equals, assert_not_equals
from nose.tools import assert_true, assert_raises
import numpy as np
from . import bbitminhash


class test_BbitMinHash(object):

    def __init__(self):
        self.bbmh = bbitminhash.BbitMinHash()

    def test__determine_dtype(self):
        pairs = ((64, np.int64), (63, np.int64), (33, np.int64),
                 (32, np.int32), (31, np.int32), (17, np.int32),
                 (16, np.int16), (15, np.int16), (9, np.int16),
                 (8, np.int8), (7, np.int8), (2, np.int8),
                 (1, np.bool))
        for (data, desired) in pairs:
            actual = self.bbmh._determine_dtype(data)
            assert_equals(actual, desired)
        for i in (65, 0, -1):
            assert_raises(ValueError, self.bbmh._determine_dtype, i)

    def test_gen_minhash(self):
        actual = self.bbmh.gen_minhash(['love', 'me'])
        assert_equals(len(actual), self.bbmh.num_hashes)
        assert_true(isinstance(actual, np.ndarray))

        other_hash = self.bbmh.gen_minhash(['tokyo', 'sushi'])
        assert_not_equals(actual.tostring(), other_hash.tostring())

    def test_compute_similarity(self):
        lhs = np.array([0] * 128)
        rhs = np.array([0] * 128)
        actual = self.bbmh.compute_similarity(lhs, rhs)
        assert_almost_equals(actual, 1)

        rhs = np.array([1] * 128)
        actual = self.bbmh.compute_similarity(lhs, rhs)
        assert_almost_equals(actual, 0)
