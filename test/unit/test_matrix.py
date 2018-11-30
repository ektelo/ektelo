import numpy as np
from ektelo import matrix, workload
import unittest

class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(10)
        n = 8
        I = matrix.Identity(n)
        O = matrix.Ones(n, n)
        P = workload.Prefix(n)
        V = matrix.VStack([I, -0.5*O])
        H = matrix.HStack([I, -0.5*O])
        S = matrix.Sum([I, -0.5*O])
        K = matrix.Kronecker([I,V,P])
        W = matrix.Weighted(K, 3.0)
        self.matrices = [I,O,P,V,H,K,O,W]
 
    def test_matmat(self):
        for Q in self.matrices:
            x = self.prng.rand(Q.shape[1])
            np.testing.assert_allclose(Q.dot(x), Q.matrix.dot(x))
            y = self.prng.rand(Q.shape[0])
            np.testing.assert_allclose(Q.T.dot(y), Q.matrix.T.dot(y))

    def test_multiplication(self):
        for Q in self.matrices:
            x = self.prng.rand(Q.shape[1])
            y = self.prng.rand(Q.shape[0])
            
            a = Q.dot(x)
            b = Q * x
            c = Q @ x 
            d = Q.matrix.dot(x)

            e = Q.T.dot(y)
            f = Q.T * y
            g = Q.T @ y
            h = Q.matrix.T.dot(y)

            np.testing.assert_allclose(a,b)
            np.testing.assert_allclose(a,c)
            np.testing.assert_allclose(a,d)

            np.testing.assert_allclose(e,f)
            np.testing.assert_allclose(e,g)
            np.testing.assert_allclose(e,h)

    def test_gram(self):
        for Q in self.matrices:
            X = Q.dense_matrix()
            np.testing.assert_allclose(X.T.dot(X), Q.gram().dense_matrix())

    def test_sum(self):
        for Q in self.matrices:
            X = Q.dense_matrix()
            
            np.testing.assert_allclose(X.sum(), Q.sum())
            np.testing.assert_allclose(X.sum(axis=0), Q.sum(axis=0))
            np.testing.assert_allclose(X.sum(axis=1), Q.sum(axis=1))

    def test_abs(self):
        for Q in self.matrices:
            A = Q.__abs__().dense_matrix()
            B = Q.dense_matrix().__abs__() 
            np.testing.assert_allclose(A,B)

if __name__ == '__main__':
    unittest.main()
