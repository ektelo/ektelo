from ektelo.algorithm.dawa.partition_engines import l1partition
from ektelo import support
from ektelo.operators import MapperOperator
import numpy as np

class Dawa(MapperOperator):

    stability = 1

    def __init__(self, eps, ratio, approx=False):
        super(Dawa, self).__init__()

        self.eps = eps
        self.ratio = ratio
        self.approx = approx

    def mapping(self, X, prng):
        pseed = prng.randint(500000)

        if self.approx:
            cluster = l1partition.L1partition_approx(X, self.eps, ratio=self.ratio, gethist=True, seed=pseed)
        else:
            cluster = l1partition.L1partition(X, self.eps, ratio=self.ratio, gethist=True, seed=pseed)

        return support.get_partition_vec(None, len(X), cluster, closeRange=True)
