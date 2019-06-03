import numpy as np
from ektelo.client.inference import LeastSquares
from ektelo.client.inference import NonNegativeLeastSquares
from ektelo.client.inference import WorkloadNonNegativeLeastSquares
from ektelo.client.inference import MultiplicativeWeights
from ektelo.client.inference import AHPThresholding
from ektelo.client.measurement import laplace_scale_factor
from ektelo.private.measurement import Laplace
from ektelo.matrix import EkteloMatrix
import unittest


class TestInference(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.eps_share = 0.1
        self.prng = np.random.RandomState(10)
        self.A = EkteloMatrix(np.eye(self.n))
        self.X = np.random.rand(self.n)

    def test_client_interaction_LS(self):
        laplace = Laplace(self.A, self.eps_share)
        ans = laplace.measure(self.X, self.prng)
        least_squares = LeastSquares()
        x_est = least_squares.infer(self.A, ans)

        self.assertEqual(self.X.shape, x_est.shape)

    def test_client_interaction_NLS(self):
        laplace = Laplace(self.A, self.eps_share)
        ans = laplace.measure(self.X, self.prng)

        non_neg_least_squares = NonNegativeLeastSquares()
        x_est = non_neg_least_squares.infer(self.A, ans)
        self.assertEqual(self.X.shape, x_est.shape)

    def test_client_interaction_WNLS(self):
        laplace = Laplace(self.A, self.eps_share)
        ans = laplace.measure(self.X, self.prng)

        engine = WorkloadNonNegativeLeastSquares(self.A)
        x_est = engine.infer(self.A, ans)
        self.assertEqual(self.X.shape, x_est.shape)

    def test_client_interaction_MW(self):
        laplace = Laplace(self.A, self.eps_share)
        ans = laplace.measure(self.X, self.prng)
        x_est_init = np.random.rand(self.n)

        mult_weight = MultiplicativeWeights()
        x_est = mult_weight.infer(self.A, ans, x_est_init)
        self.assertEqual(self.X.shape, x_est.shape)

    def test_client_interaction_HR(self):
        laplace = Laplace(self.A, self.eps_share)
        ans = laplace.measure(self.X, self.prng)
        eps_par = 0.1
        eta = 0.35
        ratio = 0.85

        AHP_threshold = AHPThresholding(eta, ratio)
        x_est = AHP_threshold.infer(self.A, ans, eps_par)
        self.assertEqual(self.X.shape, x_est.shape)

    def test_nnls(self):
        A = EkteloMatrix(np.random.rand(self.n, self.n))
        

if __name__ == '__main__':
    unittest.main()
