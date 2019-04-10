from ektelo import matrix, workload
import numpy as np
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
        M = workload.DimKMarginals((2,3,4), 2)
        D = workload.Disjuncts([P, I])
        N = workload.AllNormK(n, 2)
        self.matrices = [I,O,P,V,H,K,O,W,D,N,M]
 
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

    def test_trace(self):
        for Q in self.matrices:
            if Q.shape[0] == Q.shape[1]:
                a = Q.trace()
                b = Q.dense_matrix().trace()
                self.assertTrue(abs(a-b) <= 1e-14)

    def test_inv(self):
        for Q in self.matrices:
            if Q.shape[0] == Q.shape[1]:
                try:
                    Q1 = Q.inv().dense_matrix()
                    Q2 = np.linalg.inv(Q.dense_matrix())
                    np.testing.assert_allclose(Q1, Q2)
                except:
                    pass

    def test_pinv(self):
        for Q in self.matrices:
            Q1 = Q.pinv().dense_matrix()
            Q2 = np.linalg.pinv(Q.dense_matrix())
            #np.testing.assert_allclose(Q1, Q2, atol=1e-7)

            A = Q.dense_matrix()
            A1 = Q.pinv().dense_matrix()
            if not isinstance(Q, workload.Marginals):
                np.testing.assert_allclose(A @ A1 @ A, A, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
