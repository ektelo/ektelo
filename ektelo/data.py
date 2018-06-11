from __future__ import absolute_import
from __future__ import division
import bisect
from builtins import str
from builtins import range
from builtins import object
from collections import OrderedDict
import copy
import csv
from ektelo import util
from ektelo.mixins import Marshallable
from functools import reduce
import hashlib
import networkx as nx
import numpy as np
import math
import os
import pandas as pd
import scipy.stats as sci
import sys
import uuid
import yaml


CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')


class Node(object):

    def __init__(self):
        self.id = str(uuid.uuid4())[:8]


class Graph(object):

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_id_map = {}
        self.id_node_map = {}

        self.root = None
        self.last = None

    def insert(self, node, after=None):
        if after is None:
            after = self.last

        self._add(node)

        if after is None:
            self.root = node
        else:
            self.graph.add_edge(after.id, node.id)

        self.last = node

        return node

    def parent(self, node):
        preds = list(self.graph.predecessors(node.id))

        assert len(preds) <= 1

        if len(preds) == 1:
            return self.id_node_map[preds[0]]
        else: 
            return None

    def children(self, node):
        sucs = self.graph.successors(node.id)

        if len(sucs) == 0:
            return None
        else:
            return [self.id_node_map[suc] for suc in sucs]

    def _add(self, node):
        self.graph.add_node(node.id)
        self.node_id_map[node] = node.id
        self.id_node_map[node.id] = node


class DataManager(object):

    def __init__(self, operator):
        self._dgraph = DataGraph()
        self._dgraph.insert(operator)

    def partition(self, operator, prng, node):
        assert util.contains_superclass(operator.__class__, 'SplitByPartition')

        mapping = operator.mapping

        return [self._dgraph.insert_subset(operator, idx, node) for idx in sorted(set(mapping))]

    def transform(self, operator, node):
        assert util.contains_superclass(operator.__class__, 'TransformationOperator')

        return self._dgraph.insert(operator, node)

    def materialize(self, X, prng, node):
        if self._dgraph.root.id == node.id:
            return X, None

        paths = [path for path in nx.all_simple_paths(self._dgraph.graph, self._dgraph.root.id, node.id)]

        assert len(paths) == 1

        path = paths[0]

        for i in range(len(path)):
            node = self._dgraph.id_node_map[path[i]]

            if util.contains_superclass(node.operator.__class__, 'SplitByPartition'):
                id_node_map = self._dgraph.id_node_map

                Xs = node.operator.transform(X)
                X = Xs[node.idx]
            else:
                X = node.operator.transform(X)
     
        return X, path


class DataGraph(Graph):

    def insert(self, operator, after=None):
        node = DataNode(operator)
        super(DataGraph, self).insert(node, after)

        return node

    def insert_subset(self, operator, idxs, after=None):
        node = SubsetNode(operator, idxs)
        super(DataGraph, self).insert(node, after)

        return node


class DataNode(Node):

    def __init__(self, operator):
        super(DataNode, self).__init__()
        self.operator = operator


class SubsetNode(DataNode):

    def __init__(self, operator, idx):
        super(SubsetNode, self).__init__(operator)
        self.idx = idx


class RelationHelper(object):

    resource_csv_map = {'CPS': 'cps.csv',
                        'STROKE_1D': 'stroke.csv',
                        'STROKE_2D': 'stroke.csv'}
    resource_config_file_map = {'CPS': 'cps.yml',
                                'STROKE_1D': 'stroke.yml',
                                'STROKE_2D': 'stroke.yml'}
    resource_config_name_map = {'CPS': 'cps_config',
                                'STROKE_1D': 'stroke_1D_config',
                                'STROKE_2D': 'stroke_2D_config'}

    def __init__(self, resource, delimiter=','):
        self.delimiter = delimiter
        self.filename = os.path.join(CSV_PATH, self.resource_csv_map[resource])
        config_file = os.path.join(CONFIG_PATH, self.resource_config_file_map[resource])
        all_configs = yaml.load(open(config_file, 'r').read())
        config_name = self.resource_config_name_map[resource]
        self.config = OrderedDict(sorted(all_configs[config_name].items()))

    def load(self):
        return Relation(self.config).load_csv(self.filename, self.delimiter)


class Histogram(Marshallable):

    def __init__(self, df, bins, domains, normed=False, weights=None):
        self.init_params = util.init_params_from_locals(locals())

        self.df = df
        self.bins = bins
        self.domains = domains
        self.normed = normed
        self.weights = weights
        self.hist = None
        self.edges = None

    def generate(self):
        self.hist, self.edges = np.histogramdd(self.df.values, self.bins, self.domains, self.normed, self.weights)
        self.hist = self.hist.astype(dtype=np.int32)

        return self

    def size(self):
        return reduce(lambda x, y: x*y, self.bins)

    def statistics(self):
        assert self.hist is not None
        assert self.edges is not None

        hist_data = {}
        hist_data['nz_perc'] = util.old_div(np.count_nonzero(self.hist),float(self.hist.size))
        hist_data['max_bin_val'] = self.hist.max()
        hist_data['total_records'] = self.hist.sum()

        return hist_data


class Relation(Marshallable):

    def __init__(self, config):
        self.init_params = util.init_params_from_locals(locals())

        self.config = config
        self._df = None
        self.hist = None
        self.edges = None

    @property
    def bins(self):
        return [self.config[column]['bins'] for column in self.config]

    @property
    def domains(self):
        return [self.config[column]['domain'] for column in self.config]

    @property
    def value_map(self):
        return [(self.config[column]['value_map'] if 'value_map' in self.config[column] else None) for column in self.config]

    @property
    def df(self):
        return self._df[[column for column in self.config]]

    def load_csv(self, csv_filename, delimiter=','):
        self._df = pd.read_csv(csv_filename, sep=delimiter)
        self._apply_config()

        return self

    def where(self, query):
        assert self._df is not None, 'no data to filter'

        self._df = self._df.query(query)
        self._apply_config()

        return self

    def project(self, fields):
        all_fields = set(self.config.keys())

        assert all_fields.issuperset(set(fields)), 'supplied fields are not subset of config fields'

        self._df = self._df[fields]
        config_pairs = [(key, value) for key, value in self.config.items() if key in fields]
        self.config = OrderedDict(config_pairs)
        self._apply_config()

        return self

    def clone(self):
        return copy.deepcopy(self)

    def _apply_config(self):
        # swap categorical for numerical values
        for column in self._df.columns:
            if column in self.config and 'value_map' in self.config[column]:
                for source, target in list(self.config[column]['value_map'].items()):
                    self._df.replace({column: {source: target}}, inplace=True)

        # infer active domains
        for column in self.config:
            if self.config[column]['domain'] == 'active':
                self.config[column]['domain'] = (self._df[column].min(), self._df[column].max())
 
        return self

    def statistics(self):
        assert self._df is not None

        relation_data = {}
        relation_data['avgs'] = [self._df[column].mean() for column in self.config]
        relation_data['value_map'] = self.value_map

        return relation_data
