import numpy as np
from ektelo import matrix, workload
from ektelo.hdmm import templates
import unittest

class TestInference(unittest.TestCase):

    def setUp(self):
        self.n = 32

    def test_1d_templates(self):
        W = workload.Prefix(self.n)
        default = templates.Default(self.n, self.n)
        pid = templates.PIdentity(2, self.n)
        aug = templates.RangeTemplate(self.n, start=8, branch=2)
        eye = templates.Identity(self.n)
        eyetot = templates.IdTotal(self.n)
        x = np.random.randint(0, 100, self.n)
       
        for template in [default, pid, aug, eye, eyetot]:
            template.optimize(W)
            A = template.strategy()
            y = A.dot(x)

    def test_2d_templates(self):
        W1 = workload.Prefix2D(self.n)
        W2 = matrix.VStack([W1,W1])
        kron = templates.KronPIdentity([2,2], [self.n, self.n])
        marg = templates.Marginals([self.n, self.n])
        x = np.random.randint(0, 100, self.n**2)
       

        for W in [W1, W2]: 
            for template in [kron, marg]:
                template.optimize(W)
                A = template.strategy()
                y = A.dot(x) 

if __name__ == '__main__':
    unittest.main()
