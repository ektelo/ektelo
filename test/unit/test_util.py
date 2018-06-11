from collections import OrderedDict
from ektelo import util
from ektelo.algorithm import ahp
import json
import numpy as np
import unittest


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.A = ahp.ahpND_engine()

    def test_old_div(self):
        self.assertEqual(util.old_div(1, 2), 0)
        self.assertEqual(util.old_div(2, 2), 1)
        self.assertEqual(util.old_div(3, 2), 1)
        self.assertEqual(util.old_div(1, 2.0), 0.5)

        x = np.array((1.0,))
        y = np.array((2.0,), dtype=np.int_)
        z = np.array((2.0,), dtype=np.float_)
        w = 1.0
        zero = np.zeros((1,))
        half = 0.5 * np.ones((1,))

        self.assertEqual(util.old_div(x, y), zero)
        self.assertEqual(util.old_div(x, z), half)
        self.assertEqual(util.old_div(w, z), np.array(half))

    def test_json_primitives(self):
        self.assertEqual(13, serde(13))
        self.assertEqual(True, serde(True))
        self.assertEqual(3.14, serde(3.14))
        self.assertEqual('test', serde('test')) 
        self.assertEqual(u'test', serde(u'test')) 
        self.assertEqual(OrderedDict({'a': 1}), 
                         serde(OrderedDict({'a': 1})))
        self.assertEqual(util.standardize((3,2)), 
                         util.standardize(serde((3,2))))
        self.assertEqual(['a', 'b'], serde(['a', 'b']))
        self.assertEqual(None, serde(None))

    def test_json_numpy_array(self):
        self.assertEqual(util.standardize(np.ones((3,2))), 
                         util.standardize(serde(np.ones((3,2)))))

    def test_json_persistable(self):
        self.assertEqual(self.A.hash, serde(self.A).hash)

    def test_json_mixed(self):
        d = {'a': 13,
             5: 3.14,
             (1,2): np.ones((3,2))}

        self.assertEqual(util.standardize(d), util.standardize(serde(d)))
                          
    def test_json_hierarchical(self):
        d1 = {'a': self.A,
              'b': {'b1': self.A,
                    'b2': {'b21': self.A}},
              'c': [self.A]}

        self.assertEqual(d1['a'].hash, serde(d1)['a'].hash)
        self.assertEqual(d1['b']['b1'].hash, serde(d1)['b']['b1'].hash)
        self.assertEqual(d1['b']['b2']['b21'].hash, serde(d1)['b']['b2']['b21'].hash)
        self.assertEqual(d1['c'][0].hash, serde(d1)['c'][0].hash)

        d2 = {1: np.ones((3,2)),
              2: {21: np.ones((5,4)),
                  22: {221: np.ones((13,2))}}}

        self.assertEqual(util.standardize(d2), util.standardize(serde(d2)))

def serde(item):
    return util.receive_from_json(json.loads(json.dumps(util.prepare_for_json(item))))
