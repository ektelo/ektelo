import numpy as np
from ektelo.private.measurement import Laplace
import unittest


class TestMeasurement(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.eps_share = 0.1 
        self.seed = 10
        self.A = np.ones(self.n)
        self.X = np.ones(self.n)

    def test_laplace_operator(self):
        laplace = Laplace(self.A, self.eps_share)
        actual_meas = laplace.measure(self.X, np.random.RandomState(self.seed))

        prng = np.random.RandomState(self.seed)
        target_meas = self.n + prng.laplace(0.0, self.n/self.eps_share, self.n)

        np.testing.assert_array_equal(target_meas, actual_meas)
