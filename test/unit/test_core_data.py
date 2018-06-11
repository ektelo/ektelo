from collections import OrderedDict
from ektelo.dataset import DatasetFromRelation
from ektelo import data
import numpy as np
import pandas as pd
from ektelo import workload
from ektelo.algorithm import ahp
import os
import unittest
import yaml


csv_path = os.environ['EKTELO_DATA']
config_path = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')
config = yaml.load(open(os.path.join(config_path, 'cps.yml'), 'r').read())


class TestData(unittest.TestCase):

    def setUp(self):
        super(TestData, self).setUp()

        domain_alt = (10, 10, 7, 4, 2)   # numpy shape tuple
        self.expr_seed = 12345
        self.expr_eps = 0.1
        self.delimiter = ','

        relation = data.RelationHelper('CPS').load()

        # Full bin dataset
        self.X1 = DatasetFromRelation(relation, 'CPS-CSV')

        # Reduced bin dataset
        self.X2 = DatasetFromRelation(relation, 'CPS-CSV', reduce_to_dom_shape=domain_alt)

        # Workload and Algorithms
        self.W1 = workload.Prefix1D(20)
        self.W2 = workload.RandomRange(None, (64,), 25)
        self.A2 = ahp.ahpND_engine()

    def tearDown(self):
        super(TestData, self).tearDown()

    def test_X_can_marshal(self):
        self.assertEqual(type(self.X1.asDict()), dict)
        self.assertEqual(type(self.X2.asDict()), dict)

    def test_W_can_marshal(self):
        self.assertEqual(type(self.W1.asDict()), dict)
        self.assertEqual(type(self.W2.asDict()), dict)

    def test_A_can_marshal(self):
        self.assertEqual(type(self.A2.asDict()), dict)


if __name__ == "__main__":
    unittest.main(verbosity=3)
