from ektelo.matrix import EkteloMatrix, Identity, Ones, VStack, Kronecker
import collections
import itertools
import numpy as np
from scipy import sparse

def Total(n, dtype=np.float64):
    """
    The 1 x n matrix of 1s
    :param n: the domain size
    :return: the query matrix
    """
    return Ones(1,n,dtype)

def IdentityTotal(n, weight=1.0, dtype=np.float64):
    """
    The matrix [I; w*T] where w is the weight on the total query
    :param n: the domain size
    :param weight: the weight on the total query
    :return: the query matrix
    """
    I = Identity(n, dtype)
    T = Total(n, dtype)
    w = dtype(weight)
    return VStack([I, w*T])

class Prefix(EkteloMatrix):
    """
    The prefix workload encodes range queries of the form [0,k] for 0 <= k <= n-1
    """
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n,n)
        self.dtype = dtype

    def _matmat(self, V):
        return np.cumsum(V, axis=0)

    def _transpose(self):
        return Suffix(self.n) 

    @property
    def matrix(self):
        return np.tril(np.ones((self.n, self.n), self.dtype))
    
    def gram(self):
        y = 1 + np.arange(self.n).astype(self.dtype)[::-1]
        return EkteloMatrix(np.minimum(y, y[:,None]))
    
    def __abs__(self):
        return self

class Suffix(EkteloMatrix):
    """
    The suffix workload encodes range queries of the form [k, n-1] for 0 <= k <= n-1
    """
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n,n)
        self.dtype = dtype

    def _matmat(self, V):
        return np.cumsum(V[::-1], axis=0)[::-1]
    
    def _transpose(self):
        return Prefix(self.n)
    
    @property
    def matrix(self):
        return np.triu(np.ones((self.n, self.n), self.dtype))
    
    def gram(self):
        y = 1 + np.arange(self.n).astype(self.dtype)
        return EkteloMatrix(np.minimum(y, y[:,None]))

    def __abs__(self):
        return self

class AllRange(EkteloMatrix):
    """
    The AllRange workload encodes range queries of the form [i,j] for 0 <= i <= j <= n-1
    """
    def __init__(self, n, dtype=np.float64):
        # note: transpose is not implemented, but it's not really needed anyway
        self.n = n
        self.shape = ((n*(n+1) // 2), n)
        self.dtype = dtype
        self._prefix = Prefix(n, dtype)
        
    def _matmat(self, V):
        # probably not a good idea to ever call this function
        # should use gram when possible because this workload is so large
        # not gonna bother with a super efficient vectorized implementation
        m = self.shape[0]
        n = V.shape[1]
        ans = np.vstack([np.zeros(n), self._prefix.dot(V)])
        res = np.zeros((m, n))
        for i, (a, b) in enumerate(itertools.combinations(range(self.n+1), 2)):
            res[i] = ans[b] - ans[a]
        return res
    
    @property
    def matrix(self):
        return self.dot(np.eye(self.n))

    def gram(self):
        r = np.arange(self.n) + 1
        X = np.outer(r, r[::-1])
        return EkteloMatrix(np.minimum(X, X.T))

class RangeQueries(EkteloMatrix):
    """
    This class can represent a workload of range queries, which are provided as input
    to the constructor as a list of pairs.
    """
    def __init__(self, domain, ranges, dtype=np.float64):
        """
        :param domain: the domain size, as an int for 1D or tuple for nD domains
        :param ranges: a list of range queries, represented as a pair (lower bound, upper bound)
            where each bound is a tuple with the same size as domain.
        """
        if type(domain) is int:
            domain = (domain,)
            ranges = [( (lb,),  (ub,) ) for lb, ub in ranges]
        self.domain = domain
        self.ranges = ranges
        self.shape = (len(ranges), np.prod(domain))
        self.dtype = dtype

    @property
    def matrix(self):
        idx = np.arange(np.prod(self.domain), dtype=int).reshape(self.domain)
        row_ind = []
        col_ind = []
        for i, (lb, ub) in enumerate(self.ranges):
            s = tuple(slice(a,b+1) for a, b in zip(lb, ub))
            j = idx[s].flatten()
            col_ind.append(j)
            row_ind.append(np.repeat(i, j.size))
        row_ind = np.concatenate(row_ind)
        col_ind = np.concatenate(col_ind)
        data = np.ones_like(row_ind)
        return sparse.csr_matrix((data, (row_ind, col_ind)), self.shape)

class RandomRange(RangeQueries):
    def __init__(self, shape_list, domain, size, seed=9001):
        if type(domain) is int:
            domain = (domain,)
        self.seed = seed
        self.size = size

        prng = np.random.RandomState(seed)
        queries = []

        for i in range(size):
            if shape_list is None:
                shape = tuple(prng.randint(1, dim+1, None) for dim in domain)
            else:
                shape = shape_list[np.random.randint(len(shape_list))]
            lb = tuple(prng.randint(0, d - q + 1, None) for d,q in zip(domain, shape))
            ub = tuple(sum(x)+1 for x in zip(lb, shape))
            queries.append( (lb, ub) )

        super(RandomRange, self).__init__(domain, queries) 

class Marginal(Kronecker):
    def __init__(self, domain, binary):
        self.binary = binary
        subs = []
        for i,n in enumerate(domain):
            if binary[i] == 0:
                subs.append(Total(n))
            else:
                subs.append(Identity(n))
        Kronecker.__init__(self, subs)

class Marginals(VStack):
    def __init__(self, domain, weights):
        self.domain = domain
        self.weights = collections.defaultdict(lambda: 0.0)
        self.weights.update(weights)
        subs = []
        for key, wgt in weights.items():
            if wgt > 0: subs.append(wgt * Marginal(domain, key))
        VStack.__init__(self, subs)   

def DimKMarginals(domain, dims):
    if type(dims) is int:
        dims = [dims]
    weights = {}
    for key in itertools.product(*[[1,0]]*len(domain)):
        if sum(key) in dims:
            weights[key] = 1.0
    return Marginals(domain, weights)

def Range2D(n):
    return Kronecker([AllRange(n), AllRange(n)])

def Prefix2D(n):
    return Kronecker([Prefix(n), Prefix(n)]) 
