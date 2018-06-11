import numpy as np
import unittest
from ektelo import workload
from ektelo.client.service import ProtectedDataSource
from ektelo.plans import private


class TestWrapper(unittest.TestCase):

    def setUp(self):
        self.eps = 0.01
        self.random_seed = 10
        self.cps_domain = (10, 1, 7, 1, 1)
        self.stroke_domain = (64, 64)
        self.x_cps = ProtectedDataSource.instance('CPS', self.random_seed)
        self.x_stroke = ProtectedDataSource.instance('STROKE_2D', self.random_seed)
        self.W_cps = workload.RandomRange(None, (np.prod(self.cps_domain),), 25)
        self.W_stroke = workload.RandomRange(None, (np.prod(self.stroke_domain),), 25)

    def test_identity(self):
        x = self.x_cps.vectorize(self.cps_domain)
        private.Identity(self.cps_domain).Run(self.W_cps, x, self.eps)

    def test_h2(self):
        x = self.x_cps.vectorize(self.cps_domain)
        private.H2(self.cps_domain).Run(self.W_cps, x, self.eps)

    def test_hb(self):
        x = self.x_cps.vectorize(self.cps_domain)
        private.HB(self.cps_domain).Run(self.W_cps, x, self.eps)

    def test_hb_2D(self):
        x = self.x_stroke.vectorize(self.stroke_domain)
        private.HB(self.stroke_domain).Run(self.W_stroke, x, self.eps)

    def test_greedy_h(self):
        x = self.x_cps.vectorize(self.cps_domain)
        private.GreedyH(self.cps_domain).Run(self.W_cps, x, self.eps)

    def test_uniform(self):
        x = self.x_cps.vectorize(self.cps_domain)
        private.Uniform(self.cps_domain).Run(self.W_cps, x, self.eps)

    def test_privBayesLS(self):
        theta = 1
        private.PrivBayesLS(self.cps_domain, theta).Run(self.W_cps, self.x_cps, self.eps)

    def test_mwem(self):
        ratio = 0.5
        rounds = 3
        data_scale = 1e5
        use_history = True
        x = self.x_cps.vectorize(self.cps_domain)
        private.Mwem(self.cps_domain, ratio, rounds, data_scale, use_history).Run(self.W_cps, x, self.eps)

    def test_mwem_2D(self):
        ratio = 0.5
        rounds = 3
        data_scale = 1e5
        use_history = True
        x = self.x_stroke.vectorize(self.stroke_domain)
        private.Mwem(self.stroke_domain, ratio, rounds, data_scale, use_history).Run(self.W_stroke, x, self.eps)

    def test_ahp(self):
        eta = 0.35
        ratio = 0.85
        x = self.x_cps.vectorize(self.cps_domain)
        private.Ahp(self.cps_domain, eta, ratio).Run(self.W_cps, x, self.eps)

    def test_dawa(self):
        ratio = 0.25
        approx = False
        x = self.x_cps.vectorize(self.cps_domain)
        private.Dawa(self.cps_domain, ratio, approx).Run(self.W_cps, x, self.eps)

    def test_dawa_2D(self):
        ratio = 0.25
        approx = False
        x = self.x_stroke.vectorize(self.stroke_domain)
        private.Dawa(self.stroke_domain, ratio, approx).Run(self.W_stroke, x, self.eps)

    def test_quad_tree(self):
        x = self.x_cps.vectorize(self.cps_domain)
        private.QuadTree(self.cps_domain).Run(self.W_cps, x, self.eps)

    def test_ugrid(self):
        x_sum = 49436
        x = self.x_cps.vectorize(self.cps_domain)
        private.UGrid(self.cps_domain, x_sum).Run(self.W_cps, x, self.eps)

    def test_agrid(self):
        data_scale = 1e5
        x = self.x_stroke.vectorize(self.stroke_domain)
        private.AGrid(self.stroke_domain, data_scale).Run(self.W_stroke, x, self.eps)

    def test_dawa_striped(self):
        stripe_dim = 0
        ratio = 0.25
        approx = False
        rounds = 3
        x = self.x_cps.vectorize(self.cps_domain)
        private.DawaStriped(self.cps_domain, stripe_dim, ratio, rounds, approx).Run(self.W_cps, x, self.eps)

    def test_striped_HB_slow(self):
        stripe_dim = 0
        x = self.x_cps.vectorize(self.cps_domain)
        private.StripedHB(self.cps_domain, stripe_dim).Run(self.W_cps, x, self.eps)

    def test_mwem_variant_b(self):
        ratio = 0.5
        rounds = 3
        x = self.x_cps.vectorize(self.cps_domain)
        private.MwemVariantB(self.cps_domain, ratio, rounds).Run(self.W_cps, x, self.eps)

    def test_mwem_variant_c(self):
        ratio = 0.5
        rounds = 3
        x = self.x_cps.vectorize(self.cps_domain)
        private.MwemVariantC(self.cps_domain, ratio, rounds).Run(self.W_cps, x, self.eps)

    def test_mwem_variant_d(self):
        ratio = 0.5
        rounds = 3
        x = self.x_cps.vectorize(self.cps_domain)
        private.MwemVariantD(self.cps_domain, ratio, rounds).Run(self.W_cps, x, self.eps)
