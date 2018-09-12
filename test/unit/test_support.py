from ektelo import support
import numpy as np
from scipy import sparse
import unittest


class TestSupport(unittest.TestCase):

    def setUp(self):
        self.mapping = np.array([0,3,4,1,2,0,1,4,3,2])

    def test_split_rectangle(self):
        rows = 4
        cols = 3
        M = np.ones((rows, cols))

        np.testing.assert_array_equal(sum(support.split_rectangle(M, 1, 1)), M)

        np.testing.assert_array_equal(sum(support.split_rectangle(M, 2, 2)),
                                      np.array([[4, 4], [4, 4]]))

        np.testing.assert_array_equal(sum(support.split_rectangle(M, 4, 3)),
                                      np.array([[12]]))

    def test_get_partition_vec(self):
        n = 8
        rank = list(range(n))
        rank.reverse()
        cluster = [[0,4],[4,8]]

        np.testing.assert_array_equal(support.get_partition_vec(rank, n, cluster, closeRange=False),
                                      np.array([1,1,1,1,0,0,0,0]))

    def test_canonical_ordering(self):
        ordering = support.canonical_ordering(self.mapping)

        np.testing.assert_array_equal(ordering,
                                      np.array([0,1,2,3,4,0,3,2,1,4]))

    def test_reduction_matrix(self):
        M = support.reduction_matrix(self.mapping).dense_matrix()

        self.assertEqual(M.shape, (5,10))
        np.testing.assert_array_equal(np.nonzero(M)[1],
                                      np.array([0,5,3,6,4,9,1,8,2,7]))

    def test_expansion_matrix(self):
        M = support.expansion_matrix(self.mapping).dense_matrix()

        self.assertEqual(M.shape, (10,5))
        np.testing.assert_array_equal(np.nonzero(M)[1], self.mapping)

    def test_complimentary_reduction_expansion(self):
        R = support.reduction_matrix(self.mapping)
        E = support.expansion_matrix(self.mapping)
        
        np.testing.assert_array_equal((R*E).dense_matrix(), np.eye(5))

    def test_partition_matrix(self):
        idx = 1
        M = support.projection_matrix(self.mapping, idx).dense_matrix()

        np.testing.assert_array_equal(np.nonzero(M)[1],
                                      np.arange(10)[self.mapping==idx])

    def test_combine(self):
        v1 = np.arange(8)
        v2 = np.arange(4)
        combined = support.combine(v1, v2)

        self.assertEqual(combined.shape, (8,4))
        self.assertEqual(np.sum(combined), sum(np.arange(8*4)))

    def test_extract_M(self):
        W = sparse.csr_matrix(np.eye(5))
        M = support.extract_M(W)

        np.testing.assert_array_equal(M.toarray().flatten(), 
                                      np.eye(5)[0])

    def test_complement(self):
        M1 = sparse.csr_matrix(np.zeros((5,5)))
        M2 = sparse.csr_matrix(np.ones((5,5)))

        np.testing.assert_array_equal(support.complement(M1, 1).toarray(), 
                                      np.eye(5))
        np.testing.assert_array_equal(support.complement(M1, 5).toarray().flatten(), 
                                      np.ones(5))

        self.assertEqual(support.complement(M2, 1), None) 

    def test_get_subdomain_grid(self):
        mapping1 = np.array([
            [1,1,2,2],
            [1,1,2,2],
            [1,1,2,2]])
        d1 = support.get_subdomain_grid(mapping1, mapping1.shape)
        self.assertDictEqual(d1, {1: (3, 2), 2: (3, 2)})

        mapping2 = np.array([
            [1,1,2,2],
            [1,1,2,2],
            [3,3,0,0]])
        d2 = support.get_subdomain_grid(mapping2, mapping2.shape)
        self.assertDictEqual(d2, {1: (2, 2), 2: (2, 2), 3: (1,2), 0: (1,2)})

        mapping3 = np.array([
            [1,1,2,1]])
        d3 = support.get_subdomain_grid(mapping3, mapping3.shape)
        self.assertEqual(d3, None)

        mapping4 = np.array([
            [1,1,2,3],
            [1,1,1,4]])
        d4 = support.get_subdomain_grid(mapping4, mapping4.shape)
        self.assertEqual(d4, None)
