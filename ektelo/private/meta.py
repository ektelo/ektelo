import numpy as np
from ektelo.operators import TransformationOperator


class SplitByPartition(TransformationOperator):
	
    stability = 1

    def __init__(self, mapping):
        self.mapping = mapping

    def transform(self, X):
        sub_vectors = []

        for idx in sorted(set(self.mapping)):
            idxs = np.where(self.mapping==idx)[0]
            sub_vectors.append(X[idxs])

        return sub_vectors
