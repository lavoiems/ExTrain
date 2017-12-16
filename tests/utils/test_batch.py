import unittest
from extrain.utils import batch
import numpy as np


class TestBatch(unittest.TestCase):
    def test_batch(self):
        x = [np.array(list(range(100)))]
        mini_batches = batch.batch(10, x)
        self.assertEqual(len(mini_batches), 10)
        self.assertEqual(len(list(mini_batches[0])[0]), 10)
