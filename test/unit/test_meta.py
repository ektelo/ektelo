from ektelo.client.mapper import Grid
from ektelo.private.meta import SplitByPartition
import numpy as np
import unittest


class TestTransformation(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.grid_shape = 2
        self.idxs = [1,3,5]
        self.X = np.random.rand(self.n)
        self.prng = np.random.RandomState(10)

    def test_partition_operator(self):
        grid = Grid(self.n, self.grid_shape, canonical_order=False)
        mapping = grid.mapping()
        partition = SplitByPartition(mapping)
        sub_vectors = partition.transform(self.X)

        self.assertEqual(len(sub_vectors), 4)

        for i in range(4):
            np.testing.assert_array_equal(self.X[2*i:2*i+2], sub_vectors[i])
