import numpy as np
from ektelo.client import selection
from ektelo.private import pselection
from ektelo.matrix import EkteloMatrix
from scipy import sparse
import unittest


class TestSelection(unittest.TestCase):

    def setUp(self):

        self.domain_shape_1D = (16, )
        self.domain_shape_2D = (16, 16)
        self.W = EkteloMatrix(sparse.eye(16))

    def test_Identity(self):
        op_identity = selection.Identity(self.domain_shape_1D)
        queries = op_identity.select()
        np.testing.assert_array_equal(queries.dense_matrix(), np.eye(16))

    def test_Total(self):
        op_total = selection.Total(self.domain_shape_1D)
        queries = op_total.select()
        np.testing.assert_array_equal(queries.dense_matrix(), np.ones((1, 16)))

    def test_H2(self):
        op_H2_1D = selection.H2(self.domain_shape_1D)
        queries = op_H2_1D.select()

        self.assertEqual(queries.shape[0], 31)
        self.assertEqual(queries.shape[1], 16)

    def test_HB(self):
        op_HB_1D = selection.HB(self.domain_shape_1D)
        queries = op_HB_1D.select()

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 16)

        op_HB_2D = selection.HB(self.domain_shape_2D)
        queries = op_HB_2D.select()

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 256)

    def test_GreedyH(self):
        op_greedyH = selection.GreedyH(self.domain_shape_1D, self.W)
        queries = op_greedyH.select()

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 16)

    def test_QuadTree(self):
        op_quad_tree = selection.QuadTree(self.domain_shape_2D)
        queries = op_quad_tree.select()

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 256)

    def test_UniformGrid(self):
        op_u_grid = selection.UniformGrid(self.domain_shape_2D, 1E7, 0.1)
        queries = op_u_grid.select()

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 256)

    def test_AdaptiveGrid(self):
        op_a_grid = selection.AdaptiveGrid(self.domain_shape_2D, np.random.randint(
            0, 1000, size=self.domain_shape_2D), 0.1)
        queries = op_a_grid.select()

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 256)

    def test_Worst_approx(self):
        x = np.random.randint(
            0, 1000, size=self.domain_shape_1D)
        x_hat = np.random.randint(
            0, 1000, size=self.domain_shape_1D)
        prng = np.random.RandomState(10)

        op_worst_approx = pselection.WorstApprox(self.W, 
                                                 [],
                                                 x_hat,
                                                 0.1)

        queries = op_worst_approx.select(x, prng)

        self.assertEqual(len(queries.shape), 2)
        self.assertEqual(queries.shape[1], 16)


if __name__ == '__main__':
    unittest.main()
