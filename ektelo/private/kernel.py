from __future__ import absolute_import
from builtins import object
from ektelo import util
from ektelo.data import DataManager
from ektelo.data import Relation
from ektelo.data import Graph
from ektelo.data import Node
from ektelo.data import RelationHelper
from ektelo.private import transformation
import numpy as np


class PrivateManager(object):

    def __init__(self, source_uri, data_config, file_delimiter=',', random_seed=0, budget=1.0):
        self.source_uri = source_uri
        self.data_config = data_config
        self.file_delimiter = file_delimiter
        self.random_seed = random_seed
        self._budget = budget
        self.prng = np.random.RandomState(random_seed)
        self._data_manager = DataManager(transformation.Null())
        self._tgraph = TransformationGraph()
        self._data = None

        data_node = self._data_manager._dgraph.root
        self._tgraph.insert_transformation(data_node, transformation.Null())

    def transform(self, operator, after=None):
        assert util.contains_superclass(operator.__class__, 'TransformationOperator')

        prev_data_node = after.data_node if after is not None else None

        data_node = self._data_manager.transform(operator, prev_data_node)
        node = self._tgraph.insert_transformation(data_node, operator, after)

        return node

    def partition(self, operator, after):
        assert util.contains_superclass(operator.__class__, 'SplitByPartition')

        data_nodes = self._data_manager.partition(operator, self.prng, after.data_node)
        meta_node = self._tgraph.insert_transformation(after.data_node, operator, after)

        child_nodes = []
        for data_node in data_nodes:
            child_node = self._tgraph.insert_transformation(data_node, 
                                                            transformation.Null(), 
                                                            meta_node)
            child_nodes.append(child_node)

        return child_nodes 

    def mapping(self, node, operator, eps):
        assert util.contains_superclass(operator.__class__, 'MapperOperator')

        self._request(node, eps)
        X = self._materialize_X(node)

        return operator.mapping(X, self.prng)

    def measure(self, node, operator, eps):
        assert util.contains_superclass(operator.__class__, 'MeasurementOperator')

        self._request(node, eps)
        X = self._materialize_X(node)

        return operator.measure(X, self.prng)

    def select(self, node, operator, eps):
        assert util.contains_superclass(operator.__class__, 'SelectionOperator')

        self._request(node, eps)
        X = self._materialize_X(node)

        return operator.select(X, self.prng)

    def graph(self):
        return self._tgraph

    def _map(self, mapping, parent_node):
        groups = []
        for idx in set(mapping.vector):
            idxs = np.where(mapping.vector==idx)[0]
            groups.append(self._tgraph.insert_transformation(transformation.Group(idxs), parent_node))

        return groups

    def _materialize_X(self, node):
        X = self._load_data(self.source_uri)
        X, _ = self._data_manager.materialize(X, self.prng, node.data_node) 

        return X

    def _load_data(self, source_uri):
        if self._data is None:
            self._data = Relation(self.data_config).load_csv(self.source_uri, self.file_delimiter)
    
        return self._data

    def _request(self, node, budget, child=None):
        parent = self._tgraph.parent(node)

        if node == self._tgraph.root:
            assert parent is None

            if node.budget_consumed + budget > self._budget:
                raise BudgetExceededError()
            else:
                node.budget_consumed += budget

        elif util.contains_superclass(node.operator.__class__, 'SplitByPartition'):
            assert parent is not None
            assert child is not None

            budget_delta = max(0, child.budget_consumed + budget - node.budget_consumed)
            self._request(parent, budget_delta, node)

            node.budget_consumed += budget_delta
        else:
            self._request(parent, node.operator.stability * budget, node)
            node.budget_consumed += budget



class TransformationNode(Node):

    def __init__(self, data_node, operator):
        super(TransformationNode, self).__init__()

        self.data_node = data_node
        self.operator = operator
        self.budget_consumed = 0


class MappingNode(Node):

    def __init__(self, data_node, operator, mapping):
        super(MappingNode, self).__init__()

        self.data_node = data_node
        self.operator = operator
        self.mapping = mapping
        self.budget_consumed = 0


class TransformationGraph(Graph):

    def insert_transformation(self, data_node, operator, after=None):
        node = TransformationNode(data_node, operator)
        super(TransformationGraph, self).insert(node, after)

        return node

    def insert_mapping(self, data_node, operator, mapping, after=None):
        node = MappingNode(data_node, operator, mapping)
        super(TransformationGraph, self).insert(node, after)

        return node


class BudgetExceededError(Exception):
    pass
