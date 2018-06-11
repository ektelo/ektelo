from ektelo.private.kernel import PrivateManager
from ektelo.private.measurement import Laplace
from ektelo.client.mapper import Grid
from ektelo.private.transformation import Reshape
import numpy as np
import unittest


class TestPrivate(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.parts = 2
        self.eps_share = 0.1 
        self.prng = np.random.RandomState(10)
        self.source_uri = 'file:///path_to_data.csv'
        self.A = np.eye(self.n)

    def test_client_interaction(self):
        manager = PrivateManager(self.source_uri, None)
        manager._load_data = lambda source_uri: np.ones((self.n,))

        n1 = manager.transform(Reshape((self.n,)))
        state = manager.measure(n1, Laplace(self.A, self.eps_share), self.eps_share)
