from ektelo.dataset import DatasetFromRelation
import numpy as np
from ektelo import support
from ektelo.operators import TransformationOperator


class Vectorize(TransformationOperator):

    stability = 1

    def __init__(self, name, normed=False, weights=None, reduced_domain=None):
        self.name = name
        self.normed = normed
        self.weights = weights
        self.reduced_domain = reduced_domain

    def transform(self, relation):
        return DatasetFromRelation(relation, 
                                   self.name, 
                                   normed=self.normed,
                                   weights=self.weights,
                                   reduce_to_dom_shape=self.reduced_domain).payload.flatten()


class Where(TransformationOperator):

    stability = 1

    def __init__(self, query):
        """The query can be any valid pandas DataFrame query string:
           for example, 'age > 30'. Note that only the domains corresponding 
           to fields that are marked as 'active' in the config will be updated.
        """
        self.query = query

    def transform(self, relation):
        new_relation = relation.clone()

        return new_relation.where(self.query)


class Project(TransformationOperator):

    stability = 1

    def __init__(self, fields):
        """The fields should be a list of strings denoting the names of
           fields from the relation that should be retained.
        """
        self.fields = fields

    def transform(self, relation):
        new_relation = relation.clone()

        return new_relation.project(self.fields)


class Group(TransformationOperator):

    stability = 1

    def __init__(self, idxs):
        self.idxs = idxs 

    def transform(self, X):
        return X[self.idxs]


class ReduceByPartition(TransformationOperator):

    stability = 1

    def __init__(self, mapping):
        self.mapping = mapping

    def transform(self, X):
        return support.reduction_matrix(self.mapping) * X


class Reshape(TransformationOperator):

    stability = 1

    def __init__(self, shape):
        self.shape = shape

    def transform(self, X):
        return X.reshape(self.shape)


class Filter(TransformationOperator):

    stability = 1

    def __init__(self, mask):
        self.mask = mask

    def transform(self, X):
        assert self.mask.shape == X.shape, 'mask must have same shape as X'

        return self.mask * X


class Null(TransformationOperator):

    stability = 1

    def __init__(self):
        pass
    
    def transform(self, X):
        return X
