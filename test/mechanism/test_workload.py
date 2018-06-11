from __future__ import print_function
from builtins import range
import numpy
from ektelo.workload import *
import unittest


class WorkloadTests(unittest.TestCase):

    def setUp(self):
        self.n_sqrt = 32
        self.oneD = (self.n_sqrt * self.n_sqrt,)
        self.oneDint = self.oneD[0]

    def testLinOps(self):
        # test that W.get_matrix('linop') produces same results as
        # W.get_matrix('sparse') and W.get_matrix('dense')
        P = Prefix1D(20)
        x = numpy.random.geometric(0.01, 20)
        WD = P.get_matrix('dense')
        WS = P.get_matrix('sparse')
        WL = P.get_matrix('linop')

        numpy.testing.assert_allclose(P.evaluate(x), WD.dot(x))
        numpy.testing.assert_allclose(WD.dot(x), WS.dot(x))
        numpy.testing.assert_allclose(WD.dot(x), WL.dot(x))
        numpy.testing.assert_allclose(WD.T.dot(x), WS.T.dot(x))
        numpy.testing.assert_allclose(WD.T.dot(x), WL.H.dot(x))

    def testPrefix1D(self):

        P = Prefix1D(self.oneDint)
        PP = eval( P.__repr__() )
        self.assertEqual(P.hash, PP.hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
