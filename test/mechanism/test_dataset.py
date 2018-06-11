
"""Unit test for dataset_new.py"""
from __future__ import division
from __future__ import print_function

from builtins import range
import numpy
from ektelo import data
from ektelo import dataset
from ektelo import util
import unittest

dnames = [ 'ADULT', 'HEPTH', 'INCOME', 'MEDCOST', 'NETTRACE', 'PATENT', 'SEARCHLOGS']

class DatasetTests(unittest.TestCase):


    def setUp(self):
        n = 1024
        scale = 1E5
        self.hist = numpy.array( list(range(n)))
        self.d = dataset.Dataset(self.hist, None)
        self.dist = numpy.random.exponential(1,n)
        self.dist = util.old_div(self.dist, float(self.dist.sum()))
        self.ds = dataset.DatasetSampled(self.dist, scale, None, 1001)

    def testDataset(self):
        self.assertEqual( self.hist.sum(), self.d.scale)
        self.assertEqual( self.hist.max(), self.d.maxCount)
        print(self.d.asDict())


    def testDatasetReduce(self):
        div = 4
        new_shape = (util.old_div(self.hist.shape[0],div),)
        dr = dataset.Dataset(hist=self.hist, reduce_to_domain_shape=new_shape)
        self.assertEqual(dr.domain_shape,new_shape)

    def testDataset2D(self):
        self.X2 = dataset.DatasetFromRelation(data.RelationHelper('STROKE_2D').load(), 'STROKE', reduce_to_dom_shape=(32,32))

    def testDatasetSampled(self):
        print(self.ds.asDict())

    def testHighDimReduce(self):
        X1 = dataset.DatasetFromRelation(data.RelationHelper('CPS').load(), 'CPS', reduce_to_dom_shape=(10, 2, 7, 2, 2)).payload
        self.assertTrue(numpy.array_equal(X1.shape, (10, 2, 7, 2, 2)))
        X2 = dataset.DatasetFromRelation(data.RelationHelper('CPS').load(), 'CPS', reduce_to_dom_shape=(5, 2, 7, 2, 2)).payload
        self.assertTrue(numpy.array_equal(X2.shape, (5, 2, 7, 2, 2)))
        self.assertTrue(X1.sum()== X2.sum())

if __name__ == "__main__":
    unittest.main(verbosity=2)   
