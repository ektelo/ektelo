import numpy as np
from ektelo.workload import Marginal, Marginals, MarginalsGram, DimKMarginals
import unittest

class TestMarginals(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(10)
        self.domain = (2,3,4)

    def check_equal(self, A, B):
        self.assertEqual(A.shape, B.shape)
        v = self.prng.rand(B.shape[1])
        x = A.dot(v)
        y = B.dot(v)
        np.testing.assert_allclose(x, y)

    def test_construction(self):
        M1 = Marginal(self.domain, 5)
        M2 = Marginal.frombinary(self.domain, (1,0,1))
        M3 = Marginal.fromtuple(self.domain, (0,2))
        self.check_equal(M1, M2)
        self.check_equal(M1, M3)

        
        v = np.array([0,0,1.0,0,0,2.0,0,0])
        M1 = Marginals(self.domain, v)
        M2 = Marginals.frombinary(self.domain, { (0,1,0) : 1.0, (1,0,1) : 2.0 })
        M3 = Marginals.fromtuples(self.domain, { (1,) : 1.0, (0,2) : 2.0 })
        self.check_equal(M1, M2)
        self.check_equal(M1, M3)

    def test_approximate(self):
        M1 = Marginals(self.domain, self.prng.rand(8))
        M2 = Marginals.approximate(M1)
        np.testing.assert_allclose(M1.weights, M2.weights) 
        
        M1 = MarginalsGram(self.domain, self.prng.rand(8))
        M2 = MarginalsGram.approximate(M1)
        np.testing.assert_allclose(M1.weights, M2.weights) 

    def test_pinv(self):
        A = MarginalsGram(self.domain,  self.prng.rand(8))
        B = A.pinv()

        self.check_equal(A @ B @ A, A)

        w = self.prng.rand(8)
        w[-1] = 0  
        w = np.array([0.0,0.0,0.0,1.0,0.0,1.0,1.0,0])
        A = MarginalsGram(self.domain,  w)
        B = A.pinv()
        self.check_equal(A @ B @ A, A)

        A = Marginals(self.domain, self.prng.rand(8))
        B = A.pinv()
        self.check_equal(A @ B @ A, A)

        #B = A.gram().pinv().dense_matrix()
        #A = A.dense_matrix()
        #B = B @ A.T
        #np.testing.assert_allclose(A @ B @ A, A)

        #np.testing.assert_allclose(np.linalg.pinv(A.dense_matrix()), B.dense_matrix())

if __name__ == '__main__':
    unittest.main()
