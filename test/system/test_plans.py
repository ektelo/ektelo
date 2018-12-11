from collections import OrderedDict
from ektelo import workload
from ektelo.data import Relation
from ektelo.data import RelationHelper
import numpy as np
import os
from ektelo.plans import standalone
from ektelo.private import transformation
import unittest
import yaml


class TestPlans(unittest.TestCase):

    seed = 10
    relation_cps = RelationHelper('CPS').load()
    relation_stroke = RelationHelper('STROKE_2D').load()
    cps_domain = (10, 2, 7, 2, 2)
    stroke_domain = (64, 64)
    x_cps = transformation.Vectorize('CPS', reduced_domain=cps_domain).transform(relation_cps)
    x_cps_scale = x_cps.sum()
    x_stroke = transformation.Vectorize('STROKE', reduced_domain=stroke_domain).transform(relation_stroke)
    W_cps = workload.RandomRange(None, len(x_cps), 25)
    W_stroke = workload.RandomRange(None, len(x_stroke), 25)

    def setUp(self):
        self.eps = 0.1

    def test_identity(self):
        x_hat = standalone.Identity().Run(self.W_cps,
                                          self.x_cps,
                                          self.eps,
                                          self.seed)
        self.W_cps.dot(x_hat)

    def test_privelet(self):
        x_hat = standalone.Privelet().Run(self.W_stroke,
                                          self.x_stroke,
                                          self.eps,
                                          self.seed)
        self.W_stroke.dot(x_hat)

    def test_h2(self):
        x_hat = standalone.H2(self.cps_domain).Run(self.W_cps,
                                    self.x_cps,
                                    self.eps,
                                    self.seed)
        self.W_cps.dot(x_hat)

    def test_hb(self):
        domain_shape = (len(self.x_cps),)
        x_hat = standalone.HB(domain_shape).Run(self.W_cps,
                                                self.x_cps,
                                                self.eps,
                                                self.seed)
        self.W_cps.dot(x_hat)

    def test_hb_2D(self):
        x_hat = standalone.HB(self.stroke_domain).Run(self.W_stroke,
                                                      self.x_stroke,
                                                      self.eps,
                                                      self.seed)
        self.W_stroke.dot(x_hat)

    def test_greedy_h(self):
        x_hat = standalone.GreedyH().Run(self.W_cps,
                                         self.x_cps,
                                         self.eps,
                                         self.seed)
        self.W_cps.dot(x_hat)

    def test_uniform(self):
        x_hat = standalone.Uniform().Run(self.W_cps,
                                         self.x_cps,
                                         self.eps,
                                         self.seed)
        self.W_cps.dot(x_hat)

    def test_privBayesLS(self):
        theta = 1
        x_hat = standalone.PrivBayesLS(theta, self.cps_domain).Run(self.W_cps,
                                                                   self.relation_cps,
                                                                   self.eps,
                                                                   self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem(self):
        ratio = 0.5
        rounds = 3
        data_scale = 1e5
        domain_shape = (len(self.x_cps),)
        use_history = True
        x_hat = standalone.Mwem(ratio, 
                                rounds,
                                data_scale,
                                domain_shape,
                                use_history).Run(self.W_cps,
                                                 self.x_cps,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_2D(self):
        ratio = 0.5
        rounds = 3
        data_scale = 1e5
        use_history = True
        x_hat = standalone.Mwem(ratio, 
                                rounds,
                                data_scale,
                                self.stroke_domain,
                                use_history).Run(self.W_stroke,
                                                 self.x_stroke,
                                                 self.eps,
                                                 self.seed)
        self.W_stroke.dot(x_hat)

    def test_ahp(self):
        eta = 0.35
        ratio = 0.85
        x_hat = standalone.Ahp(eta, ratio).Run(self.W_cps,
                                               self.x_cps,
                                               self.eps,
                                               self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa(self):
        ratio = 0.25
        approx = False
        domain_shape = (len(self.x_cps),)
        x_hat = standalone.Dawa(domain_shape, ratio, approx).Run(self.W_cps,
                                                                 self.x_cps,
                                                                 self.eps,
                                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa_2D(self):
        ratio = 0.25
        approx = False
        x_hat = standalone.Dawa(self.stroke_domain, ratio, approx).Run(self.W_stroke,
                                                                       self.x_stroke,
                                                                       self.eps,
                                                                       self.seed)
        self.W_stroke.dot(x_hat)

    def test_quad_tree(self):
        x = self.x_cps.reshape((len(self.x_cps) // 2, 2))
        x_hat = standalone.QuadTree().Run(self.W_cps,
                                          x,
                                          self.eps,
                                          self.seed)
        self.W_cps.dot(x_hat)

    def test_ugrid(self):
        data_scale = 1e5
        x = self.x_cps.reshape((len(self.x_cps) // 2, 2))
        x_hat = standalone.UGrid(data_scale).Run(self.W_cps,
                                                 x,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_agrid(self):
        data_scale = 1e5
        x = self.x_cps.reshape((len(self.x_cps) // 2, 2))
        x_hat = standalone.AGrid(data_scale).Run(self.W_cps,
                                                 x,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_agrid_fast(self):
        data_scale = 1e5
        x = self.x_cps.reshape((len(self.x_cps) // 2, 2))
        x_hat = standalone.AGrid_fast(data_scale).Run(self.W_cps,
                                                 x,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa_striped(self):
        stripe_dim = 0
        ratio = 0.25
        approx = False
        x_hat = standalone.DawaStriped(self.cps_domain, stripe_dim, ratio, approx).Run(self.W_cps,
                                                                                       self.x_cps,
                                                                                       self.eps,
                                                                                       self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa_striped_fast(self):
        stripe_dim = 0
        ratio = 0.25
        approx = False
        x_hat = standalone.DawaStriped_fast( self.cps_domain, stripe_dim, ratio, approx).Run(self.W_cps,
                                                                                       self.x_cps,
                                                                                       self.eps,
                                                                                       self.seed)
        self.W_cps.dot(x_hat)

    def test_striped_HB_slow(self):
        stripe_dim = 0
        x_hat = standalone.StripedHB(self.cps_domain, stripe_dim).Run(self.W_cps,
                                                                      self.x_cps,
                                                                      self.eps,
                                                                      self.seed)
        self.W_cps.dot(x_hat)

    def test_striped_HB_fast(self):
        stripe_dim = 0
        x_hat = standalone.StripedHB_fast(self.cps_domain, 'MM', stripe_dim).Run(self.W_cps,
                                                                      self.x_cps,
                                                                      self.eps,
                                                                      self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_variant_b(self):
        ratio = 0.5
        rounds = 3
        x_hat = standalone.MwemVariantB(ratio, rounds, self.x_cps_scale, self.cps_domain, True).Run(self.W_cps,
                                                           self.x_cps,
                                                           self.eps,
                                                           self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_variant_c(self):
        ratio = 0.5
        rounds = 3
        total_noise_scale = 30
        x_hat = standalone.MwemVariantC(ratio, rounds, self.x_cps_scale, self.cps_domain, total_noise_scale).Run(self.W_cps,
                                                           self.x_cps,
                                                           self.eps,
                                                           self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_variant_d(self):
        ratio = 0.5
        rounds = 3
        total_noise_scale = 30
        x_hat = standalone.MwemVariantD(ratio, rounds, self.x_cps_scale, self.cps_domain, total_noise_scale).Run(self.W_cps,
                                                           self.x_cps,
                                                           self.eps,
                                                           self.seed)
        self.W_cps.dot(x_hat)
