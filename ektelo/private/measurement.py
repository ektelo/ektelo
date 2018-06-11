from __future__ import division
import math
import numpy as np
from ektelo import util
from ektelo.operators import MeasurementOperator


class Laplace(MeasurementOperator):

    def __init__(self, A, eps):
        self.A = A
        self.eps = eps

    def measure(self, X, prng):
        sensitivity = self.sensitivity_L1(self.A)
        laplace_scale = util.old_div(sensitivity, float(self.eps))
        noise = prng.laplace(0.0, laplace_scale, self.A.shape[0])

        return self.A.dot(X) + noise

    @staticmethod
    def sensitivity_L1(A):
        """Return the L1 sensitivity of input matrix A: maximum L1 norm of the columns."""
        return float(np.abs(A).sum(axis=0).max()) # works efficiently for both numpy arrays and scipy matrices
