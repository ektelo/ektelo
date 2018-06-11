from ektelo.private import kernel
from ektelo.private import measurement
from ektelo.private import meta
from ektelo.private import pmapper
from ektelo.private import pselection
from ektelo.private import transformation
import numpy as np


class KernelService:

    def __init__(self, private_manager):
        self.private_manager = private_manager

    def partition(self, node_id, mapping):
        node = self.private_manager.graph().id_node_map[node_id]
        part = meta.SplitByPartition(mapping)
        private_nodes = self.private_manager.partition(part, after=node)

        return [private_node.id for private_node in private_nodes]

    def transform(self, node_id, class_name, init_params):
        node = self.private_manager.graph().id_node_map[node_id]
        trans = getattr(transformation, class_name)(**init_params)
        private_node = self.private_manager.transform(trans, after=node)

        return private_node.id

    def measure(self, node_id, class_name, eps, init_params):
        node = self.private_manager.graph().id_node_map[node_id]
        msmt = getattr(measurement, class_name)(**init_params)

        return self.private_manager.measure(node, msmt, eps)

    def mapping(self, node_id, class_name, eps, init_params):
        node = self.private_manager.graph().id_node_map[node_id]
        mpr = getattr(pmapper, class_name)(**init_params)

        return self.private_manager.mapping(node, mpr, eps)

    def select(self, node_id, class_name, eps, init_params):
        node = self.private_manager.graph().id_node_map[node_id]
        slct = getattr(pselection, class_name)(**init_params)

        return self.private_manager.select(node, slct, eps)

    def get_root_id(self):
        return self.private_manager._tgraph.root.id
