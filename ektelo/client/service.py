from builtins import zip
from builtins import object
import numpy as np
from ektelo.data import DataManager
from ektelo.data import RelationHelper
from ektelo.client import inference
from ektelo.client import mapper
from ektelo.client import selection
from ektelo.private import service as pservice
from ektelo.private import kernel
from ektelo.private import measurement
from ektelo.private import meta
from ektelo.private import pmapper
from ektelo.private import pselection
from ektelo.private import transformation


class ProtectedDataSource(object):

    def __init__(self, 
                 kernel_service, 
                 data_manager=DataManager(transformation.Null()), 
                 private_node_id=None, 
                 data_node=None, 
                 random_seed=None):
        self.kernel_service = kernel_service
        self.data_manager = data_manager
        self.prng = np.random.RandomState(random_seed)

        if private_node_id is not None:
            self.private_node_id = private_node_id
        else:
            self.private_node_id = self.kernel_service.get_root_id() 

        if data_node is not None:
            self.data_node = data_node
        else:
            self.data_node = self.data_manager._dgraph.root

    @staticmethod
    def instance(source_uri, random_seed=0):
        relation_helper = RelationHelper(source_uri) 
        private_manager = kernel.PrivateManager(relation_helper.filename, 
                                                relation_helper.config, 
                                                random_seed=random_seed)
        kernel_service = pservice.KernelService(private_manager)

        return ProtectedDataSource(kernel_service,
                                   DataManager(transformation.Null()))

    def transform(self, class_name, init_params={}):
        private_node_id = self.kernel_service.transform(self.private_node_id,
                                                        class_name, 
                                                        init_params)

        data_node = self.data_manager.transform(getattr(transformation, class_name)(*init_params), 
                                                self.data_node)

        return self._clone(private_node_id, data_node)

    def measure(self, class_name, eps, init_params={}):
        return self.kernel_service.measure(self.private_node_id, class_name, eps, init_params)

    def select(self, class_name, eps, init_params={}):
        return self.kernel_service.select(self.private_node_id, class_name, eps, init_params)

    def mapping(self, class_name, eps, init_params={}):
        return self.kernel_service.mapping(self.private_node_id, class_name, eps, init_params)

    def dawa(self, ratio, approx, eps):
        return self.mapping(pmapper.Dawa.__name__, eps, {'eps': eps, 'ratio': ratio, 'approx': approx})

    def ahp_partition(self, n, ratio, eta, eps):
        M = selection.Identity((n,)).select() 
        y = self.laplace(M, ratio * eps)
        xest = inference.AHPThresholding(eta, ratio).infer(M, y, eps)

        return mapper.AHPCluster(xest, (1-ratio) * eps).mapping()

    def vectorize(self, reduced_domain):
        return self.transform(transformation.Vectorize.__name__, {'name': '', 'reduced_domain': reduced_domain})

    def where(self, query):
        return self.transform(transformation.Where.__name__, {'query': query})

    def project(self, fields):
        return self.transform(transformation.Project.__name__, {'fields': fields})

    def laplace(self, M, eps):
        return self.measure(measurement.Laplace.__name__, eps, {'A': M, 'eps': eps})

    def split_by_partition(self, mapping):
        private_node_ids = self.kernel_service.partition(self.private_node_id, mapping)

        data_nodes = self.data_manager.partition(meta.SplitByPartition(mapping), 
                                                 self.prng,
                                                 self.data_node)

        return [self._clone(private_node_id, data_node) for private_node_id, data_node in zip(private_node_ids, data_nodes)]

    def reduce_by_partition(self, mapping):
        return self.transform(transformation.ReduceByPartition.__name__, {'mapping': mapping})

    def reshape(self, shape):
        return self.transform(transformation.Reshape.__name__, {'shape': shape})

    def priv_bayes_select(self, theta, domain, eps):
        return self.select(pselection.PrivBayesSelect.__name__, 
                           eps, 
                           {'theta': theta, 'domain_shape': domain, 'eps': eps})

    def worst_approx(self, W, measuredQueries, x_hat, eps, mechanism='EXPONENTIAL'):
        return self.select(pselection.WorstApprox.__name__, 
                           eps, 
                           {'W': W, 'measuredQueries': measuredQueries, 'x_est': x_hat, 'eps': eps, 'mechanism': mechanism})

    def _clone(self, private_node, data_node):
        return ProtectedDataSource(self.kernel_service,
                                   self.data_manager,
                                   private_node,
                                   data_node)
