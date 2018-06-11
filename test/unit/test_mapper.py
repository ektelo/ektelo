import numpy as np
from ektelo.private.pmapper import Dawa
from ektelo.client.mapper import Grid
from ektelo.client.mapper import Striped
import unittest


class TestMapper(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.eps = 0.1
        self.ratio = 0.25
        self.prng = np.random.RandomState(10)

    def test_grid_operator(self):
        for grid_shape in [2,4]:
            grid = Grid(self.n, grid_shape, canonical_order=False)
            mapping = grid.mapping()

            self.assertEqual(mapping.size, self.n)
            self.assertEqual(len(set(mapping)), self.n/grid_shape) 

    def test_dawa_operator(self):
        x = (self.prng.rand(self.n) * 100).astype(np.int)
        dawa = Dawa(self.eps, self.ratio, False)
        mapping = dawa.mapping(x, self.prng)

        self.assertEqual(mapping.size, 8)

    def test_striped_operator(self):
        domain = (8, 4, 2)
        n = np.prod(domain)

        for stripe_dim in range(len(domain)):
            striped = Striped(domain, stripe_dim)
            mapping = striped.mapping()

            self.assertEqual(mapping.size, n)
            self.assertEqual(len(set(mapping)), np.prod(domain) / domain[stripe_dim]) 
