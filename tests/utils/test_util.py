import unittest
from extrain.utils import util
import numpy as np


class TestUtil(unittest.TestCase):
    def test_transpose(self):
        a = [[1,2], [3, 4], [5, 6]]
        a_t = list(util.transpose(a))
        self.assertEqual(a_t, [[1, 3, 5], [2, 4, 6]])

        b = [1, 2, 3, 4]
        self.assertRaises(TypeError, util.transpose, b)

    def test_shuffle(self):
        a = np.array([1, 2, 3, 4, 5])
        a_s = util.shuffle(a)
        self.assertFalse((a == a_s).all())
