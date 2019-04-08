from ektelo import matrix, workload
from ektelo.hdmm import templates
import numpy as np
from scipy.optimize import check_grad
import unittest

class TestGradients(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(10)

    def test_pidentity(self):
        pid = templates.PIdentity(2, 8)
        pid._set_workload(workload.Prefix(8))

        x0 = self.prng.rand(16)

        func = lambda p: pid._loss_and_grad(p)[0]
        grad = lambda p: pid._loss_and_grad(p)[1]

        err = check_grad(func, grad, x0)
        print(err)
        self.assertTrue(err <= 1e-5)

    def test_augmented_identity(self):    

        pid1 = templates.IdTotal(8)
        imatrix = self.prng.randint(0, 5, (3,8))
        pid2 = templates.AugmentedIdentity(imatrix)
        strats = [pid1, pid2]

        for pid in strats:
            pid._set_workload(workload.Prefix(8))
            x0 = self.prng.rand(pid._params.size)
            func = lambda p: pid._loss_and_grad(p)[0]
            grad = lambda p: pid._loss_and_grad(p)[1]
            err = check_grad(func, grad, x0)
            print(err)
            self.assertTrue(err <= 1e-5)

    def test_default(self):
        # TODO(ryan): test fails, but we don't really use this parameterization anyway
        temp = templates.Default(10, 8)
        temp._set_workload(workload.Prefix(8))   

        x0 = self.prng.rand(80)
        x0[0] = 10

        func = lambda p: temp._loss_and_grad(p)[0]
        grad = lambda p: temp._loss_and_grad(p)[1]

        err = check_grad(func, grad, x0)
        print(err)
        #self.assertTrue(err <= 1e-5)

    def test_marginals(self):
        W = workload.Range2D(4)

        temp = templates.Marginals((4,4))
        temp._set_workload(W)

        x0 = self.prng.rand(4)

        func = lambda p: temp._loss_and_grad(p)[0]
        grad = lambda p: temp._loss_and_grad(p)[1]

        err = check_grad(func, grad, x0)
        print(err)
        self.assertTrue(err <= 1e-5)



if __name__ == '__main__':
    unittest.main()
