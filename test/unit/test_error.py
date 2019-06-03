from ektelo import error, matrix, workload
import numpy as np
import unittest

class TestError(unittest.TestCase):

    def setUp(self):
        
        self.prng = np.random.RandomState(0)
        
        self.domain = (2,3,4)
        I = lambda n: workload.Identity(n)
        T = lambda n: workload.Total(n)
        P = lambda n: workload.Prefix(n)
        R = lambda n: matrix.EkteloMatrix(self.prng.rand(n,n))
    
        W1 = workload.Kronecker([I(2), T(3), P(4)])
        W2 = workload.Kronecker([T(2), T(3), I(4)])
        W3 = workload.Kronecker([I(2), I(3), T(4)])
        
        self.W = workload.VStack([W1, W2, W3])

        # three representations of Identity matrix
        self.A1 = I(2*3*4)
        self.A2 = workload.Kronecker([I(2),I(3),I(4)])
        self.A3 = workload.Marginals.fromtuples(self.domain, {(0,1,2) : 1.0 })

        self.A4 = workload.Marginals(self.domain, self.prng.rand(8))
        self.A5 = workload.Kronecker([R(2), R(3), R(4)])

    def test_identity(self):
        ans1 = error.rootmse(self.W, self.A1)
        ans2 = error.rootmse(self.W, self.A2)
        ans3 = error.rootmse(self.W, self.A3)
        print(ans1, ans2, ans3)
        self.assertTrue(abs(ans1-ans2) <= 1e-10)
        self.assertTrue(abs(ans1-ans3) <= 1e-10)

        ans1 = error.per_query_error(self.W, self.A1)
        ans2 = error.per_query_error(self.W, self.A2)
        ans3 = error.per_query_error(self.W, self.A3)

        np.testing.assert_allclose(ans1, ans2)
        np.testing.assert_allclose(ans1, ans3)

    def test_per_query(self):
        # test that total per query error == expected error for all strategies
        for A in [self.A1, self.A2, self.A3, self.A4, self.A5]:
            e1 = error.expected_error(self.W, A)
            e2 = error.per_query_error(self.W, A).sum()
            print(e1, e2)
            self.assertTrue(abs(e1-e2) <= 1e-4)

    def test_matrix(self):
        # test that error using explicit matrices matches error using implicit representations
        for A in [self.A1, self.A2, self.A3, self.A4, self.A5]:
            e1 = error.expected_error(self.W, A)
            e2 = error.expected_error(self.W.dense_matrix(), A.dense_matrix())
            print(e1, e2)
            self.assertTrue(abs(e1-e2)/e1 <= 1e-7)
 
if __name__ == '__main__':
    unittest.main()
