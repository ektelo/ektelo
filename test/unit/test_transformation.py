from collections import OrderedDict
from ektelo.data import Relation
from ektelo.data import RelationHelper
import numpy as np
import os
from ektelo.client.mapper import Grid
from ektelo.private.transformation import Group
from ektelo.private.transformation import Null
from ektelo.private.transformation import ReduceByPartition
from ektelo.private.transformation import Reshape
from ektelo.private.transformation import Filter
from ektelo.private.transformation import Where
from ektelo.private.transformation import Project
from ektelo.private.transformation import Vectorize
import unittest
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')


class TestTransformation(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.grid_shape = 2
        self.idxs = [1,3,5]
        self.X = np.random.rand(self.n)

        delimiter = ','
        self.reduced_domain = (10, 10, 7, 4, 2)
        config = yaml.load(open(os.path.join(CONFIG_PATH, 'cps.yml'), 'r').read())
        self.relation = RelationHelper('CPS').load()

    def test_vectorize_operator(self):
        vectorize = Vectorize('CPS-CSV', reduced_domain=self.reduced_domain)
        transformation = vectorize.transform(self.relation)
        X = transformation

        self.assertEqual(np.prod(self.reduced_domain), len(X)) 

    def test_where_operator(self):
        where = Where('age >= 30')
        X = where.transform(self.relation)

        self.assertEqual(X._df.age.min(), 30) 

    def test_project_operator(self):
        project = Project(['income'])
        X = project.transform(self.relation)

        np.testing.assert_array_equal(X.domains, [X.config['income']['domain']])

    def test_group_operator(self):
        group = Group(self.idxs)        
        transformation = group.transform(self.X)

        self.assertEqual(transformation.shape, (3,))
        np.testing.assert_array_equal(transformation, self.X[self.idxs])

    def test_reduce_operator(self):
        grid = Grid(self.n, self.grid_shape, canonical_order=False)
        mapping = grid.mapping()
        reduction = ReduceByPartition(mapping)
        transformation = reduction.transform(self.X)

        for i in range(4):
            self.assertEqual(sum(self.X[2*i:2*i+2]), transformation[i])

    def test_reshape_operator(self):
        shape = (4, 2)
        reshaper = Reshape(shape)

        x_hat = reshaper.transform(self.X)

        self.assertEqual(x_hat.shape, shape)

    def test_filter_operator(self):
        vectorize = Vectorize('CPS-CSV', reduced_domain=self.reduced_domain)
        transformation = vectorize.transform(self.relation)
        X = transformation
        mask = np.ones(self.reduced_domain).flatten()

        filterer = Filter(mask)
        self.assertEqual(sum(filterer.transform(X)), sum(X))

        filterer = Filter(1-mask)
        self.assertEqual(sum(filterer.transform(X)), 0)

    def test_null_operator(self):
        null = Null()
        transformation = null.transform(self.X)

        np.testing.assert_array_equal(transformation, self.X)
