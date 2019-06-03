from ektelo import matrix
from ektelo.matrix import EkteloMatrix, Identity, Ones, VStack, Kronecker, Sum, Weighted
import collections
import functools
import itertools
import numpy as np
from scipy.special import binom
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

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

class RangeQueries(matrix.Product):
    """
    This class can represent a workload of range queries, which are provided as input
    to the constructor.
    """
    def __init__(self, domain, lower, higher, dtype=np.float64):
        """
        :param domain: the domain size, as an int for 1D or tuple for d-dimensional 
            domains where each bound is a tuple with the same size as domain.
        :param lower: a q x d array of lower boundaries for the q queries
        :param higher: a q x d array of upper boundareis for the q queries
        """
        assert lower.shape == higher.shape, 'lower and higher must have same shape'
        #assert np.all(lower <= higher), 'lower index must be <= than higher index'

        if type(domain) is int:
            domain = (domain,)
            lower = lower[:,None]
            higher = higher[:,None]
        self.domain = domain
        self.shape = (lower.shape[0], np.prod(domain))
        self.dtype = dtype
        self._lower = lower
        self._higher = higher

        idx = np.arange(np.prod(domain), dtype=np.int32).reshape(domain)
        shape = (lower.shape[0], np.prod(domain))
        corners = np.array(list(itertools.product(*[(False,True)]*len(domain))))
        size = len(corners)*lower.shape[0]
        row_ind = np.zeros(size, dtype=np.int32)
        col_ind = np.zeros(size, dtype=np.int32)
        data = np.zeros(size, dtype=dtype)
        queries = np.arange(shape[0], dtype=np.int32) 
        start = 0
        
        for corner in corners:
            tmp = np.where(corner, lower-1, higher)
            keep = np.all(tmp >= 0, axis=1)
            index = idx[tuple(tmp.T)]
            coef = np.sum(corner)%2 * 2 - 1
            end = start + keep.sum()
            row_ind[start:end] = queries[keep]
            col_ind[start:end] = index[keep]
            data[start:end] = -coef
            start = end

        self._transformer=sparse.csr_matrix((data[:end],(row_ind[:end],col_ind[:end])),shape,dtype)

        P = Kronecker([Prefix(n, dtype) for n in domain])
        T = EkteloMatrix(self._transformer)
        matrix.Product.__init__(self, T, P)

    @staticmethod
    def fromlist(domain, ranges, dtype=np.float64):
        """ create a matrix of range queries from a list of (lower, upper) pairs
        
        :param domain: the domain of the range queries
        :param ranges: a list of (lower, upper) pairs, where 
            lower and upper are tuples with same size as domain
        """
        lower, higher = np.array(ranges).transpose([1,0,2])
        return RangeQueries(domain, lower, higher, dtype)
    
    @property
    def matrix(self):
        idx = np.arange(np.prod(self.domain), dtype=int).reshape(self.domain)
        row_ind = []
        col_ind = []
        for i, (lb, ub) in enumerate(zip(self._lower, self._higher)):
            s = tuple(slice(a,b+1) for a, b in zip(lb, ub))
            j = idx[s].flatten()
            col_ind.append(j)
            row_ind.append(np.repeat(i, j.size))
        row_ind = np.concatenate(row_ind)
        col_ind = np.concatenate(col_ind)
        data = np.ones_like(row_ind)
        return sparse.csr_matrix((data, (row_ind, col_ind)), self.shape, self.dtype)

    def __abs__(self):
        return self

    def unproject(self, offset, domain):
        return RangeQueries(domain, self._lower+np.array(offset), self._higher+np.array(offset))

class Marginal(Kronecker):
    def __init__(self, domain, key):
        """
        :param domain: a d-tuple containing the domain size of the d attributes
        :param key: a integer key 0 <= key < 2^d identifying the marginal
        """
        self.domain = domain
        self.key = key
        binary = self.binary()
        subs = []
        for i,n in enumerate(domain):
            if binary[i] == 0:
                subs.append(Total(n))
            else:
                subs.append(Identity(n))
        Kronecker.__init__(self, subs)

    def binary(self):
        i = self.key
        d = len(self.domain)
        return tuple([int(bool(2**k & i)) for k in range(d)])[::-1]

    def tuple(self):
        binary = self.binary()
        d = len(self.domain)
        return tuple(i for i in range(d) if binary[i] == 1)

    @staticmethod
    def frombinary(domain, binary):
        d = len(domain)
        key = sum(binary[k]*2**(d-k-1) for k in range(d))
        return Marginal(domain, key)

    @staticmethod
    def fromtuple(domain, attrs):
        binary = [1 if i in attrs else 0 for i in range(len(domain))]
        return Marginal.frombinary(domain, binary)

class Marginals(VStack):
    def __init__(self, domain, weights):
        self.domain = tuple(domain)
        self.weights = weights
        subs = []
        for key, wgt in enumerate(weights):
            if wgt > 0: subs.append(wgt * Marginal(domain, key))
        VStack.__init__(self, subs)   

    def gram(self):
        return MarginalsGram(self.domain, self.weights**2)
   
    def pinv(self):
        # note: this is a generalized inverse, not necessarily the pseudo inverse though
        return self.gram().pinv() * self.T

    @staticmethod 
    def frombinary(domain, weights):
        vect = np.zeros(2**len(domain))
        for binary, wgt in weights.items():
            M = Marginal.frombinary(domain, binary)
            vect[M.key] = wgt
        return Marginals(domain, vect)

    @staticmethod
    def fromtuples(domain, weights):
        vect = np.zeros(2**len(domain))
        for tpl, wgt in weights.items():
            M = Marginal.fromtuple(domain, tpl)
            vect[M.key] = wgt
        return Marginals(domain, vect)
   
    @staticmethod
    def approximate(W):
        """
        Given a Union-of-Kron workload, find a Marginals workload that approximates it.
        
        The guarantee is that for all marginals strategies A, Error(W, A) = Error(M, A) where
        M is the returned marginals approximation of W.
        The other guarantee is that this function is idempotent: approx(approx(W)) = approx(W)
        """
        WtW = MarginalsGram.approximate(W.gram())
        return Marginals(WtW.domain, np.sqrt(WtW.weights))

class MarginalsGram(Sum):
    def __init__(self, domain, weights):
        self.domain = tuple(domain)
        self.weights = weights
        subs = []
        n, d = np.prod(domain), len(domain)
        mult = np.ones(2**d)
        for key, wgt in enumerate(weights):
            Q = Marginal(domain, key)
            mult[key] = n / Q.shape[0]
            if wgt != 0: subs.append(wgt * Q.gram())
        self._mult = mult

        Sum.__init__(self, subs)

    def _Xmatrix(self,vect):
        # the matrix X such that M(u) M(v) = M(X(u) v)
        d = len(self.domain)
        A = np.arange(2**d)
        mult = self._mult

        values = np.zeros(3**d)
        rows = np.zeros(3**d, dtype=int)
        cols = np.zeros(3**d, dtype=int)
        start = 0
        for b in range(2**d):
            #uniq, rev = np.unique(a&B, return_inverse=True) # most of time being spent here
            mask = np.zeros(2**d, dtype=int)
            mask[A&b] = 1
            uniq = np.nonzero(mask)[0]
            step = uniq.size
            mask[uniq] = np.arange(step)
            rev = mask[A&b]
            values[start:start+step] = np.bincount(rev, vect*mult[A|b], step)
            cols[start:start+step] = b
            rows[start:start+step] = uniq
            start += step
        X = sparse.csr_matrix((values, (rows, cols)), (2**d, 2**d))
        XT = sparse.csr_matrix((values, (cols, rows)), (2**d, 2**d))
        return X, XT

    def __mul__(self, other):
        if isinstance(other, MarginalsGram) and self.domain == other.domain:
            X, XT = self._Xmatrix(self.weights)
            vect = X.dot(other.weights)
            return MarginalsGram(self.domain, vect)
        else:
            return EkteloMatrix.__mul__(self, other)

    def inv(self):
        assert self.weights[-1] != 0, 'matrix is not invertible'
        X, _ = self._Xmatrix(self.weights)
        z = np.zeros_like(self.weights)
        z[-1] = 1.0
        phi = spsolve_triangular(X, z, lower=False)
        return MarginalsGram(self.domain, phi)

    def ginv(self):
        w = self.weights
        X, _ = self._Xmatrix(w)
        idx = X.dot(np.ones(w.size)) != 0
        X = X[idx,:][:,idx] # to make solve_triangular work
        phi = spsolve_triangular(X, w[idx], lower=False)
        phi = spsolve_triangular(X, phi, lower=False)
        ans = np.zeros(w.size)
        ans[idx] = phi
        return MarginalsGram(self.domain, ans)

    def pinv(self):
        return self.ginv()

    def trace(self):
        return self.weights.sum() * self.shape[1]

    @staticmethod
    def approximate(WtW):
        """
        Given a Sum-of-Kron matrix, find a MarginalsGram object that approximates it.
        """
        WtW = sum_kron_canonical(WtW)
        dom = tuple(Wi.shape[1] for Wi in WtW.matrices[0].base.matrices)
        weights = np.zeros(2**len(dom))
        for sub in WtW.matrices:
            tmp = []
            for n, piece in zip(dom, sub.base.matrices):
                X = piece.dense_matrix()
                b = float(X.sum() - X.trace()) / (n * (n-1))
                a = float(X.trace()) / n - b
                tmp.append(np.array([b,a]))
            weights += sub.weight * functools.reduce(np.kron, tmp)
        return MarginalsGram(dom, weights)
 
class AllNormK(EkteloMatrix):
    def __init__(self, n, norms, dtype=np.float64):
        """
        All predicate queries that sum k elements of the domain
        :param n: The domain size
        :param norms: the L1 norm (number of 1s) of the queries (int or list of ints)
        """
        self.n = n
        if type(norms) is int:
            norms = [norms]
        self.norms = norms
        self.m = int(sum(binom(n, k) for k in norms))
        self.shape = (self.m, self.n)
        self.dtype = dtype

    @property
    def matrix(self):
        Q = np.zeros((self.m, self.n))
        idx = 0
        for k in self.norms:
            for q in itertools.combinations(range(self.n), k):
                Q[idx, q] = 1.0
                idx += 1
        return Q

    def gram(self):
        # WtW[i,i] = binom(n-1, k-1) (1 for each query having q[i] = 1)
        # WtW[i,j] = binom(n-2, k-2) (1 for each query having q[i] = q[j] = 1)
        n = self.n
        diag = sum(binom(n-1, k-1) for k in self.norms)
        off = sum(binom(n-2, k-2) for k in self.norms)
        return off*Ones(n,n) + (diag-off)*Identity(n)

class Disjuncts(Sum):
    """
    Just like the Kron workload class can represent a cartesian product of predicate counting
    queries where the predicates are conjunctions, this workload class can represent a cartesian
    product of predicate counting queries where the predicates are disjunctions.
    """
    #TODO: check implementation after refactoring
    def __init__(self, workloads):
        # q or r = - (-q and -r)
        # W = 1 x 1 - (Q1 x R1)

        self.A = Kronecker([Ones(*W.shape) for W in workloads]) # totals
        self.B = -1*Kronecker([Ones(*W.shape) - W for W in workloads]) # negations
        Sum.__init__(self, [self.A, self.B])

    def gram(self):
        return Sum([self.A.gram(), self.A.T @ self.B, self.B.T @ self.A, self.B.gram()])

class ExplicitGram:
    # not an Ektelo Matrix, but behaves like it in the sense that it has gram function,
    # meaning strategy optimization is possible
    def __init__(self, matrix):
        self.matrix = matrix
    
    def gram(self):
        return EkteloMatrix(self.matrix)

def RandomRange(shape_list, domain, size, seed=9001):
    if type(domain) is int:
        domain = (domain,)

    prng = np.random.RandomState(seed)
    queries = []

    for i in range(size):
        if shape_list is None:
            shape = tuple(prng.randint(1, dim+1, None) for dim in domain)
        else:
            shape = shape_list[np.random.randint(len(shape_list))]
        lb = tuple(prng.randint(0, d - q + 1, None) for d,q in zip(domain, shape))
        ub = tuple(sum(x)-1 for x in zip(lb, shape))
        queries.append( (lb, ub) )

    return RangeQueries.fromlist(domain, queries) 

def Moments(n, k=3):
    N = np.arange(n)
    K = np.arange(1,k+1)
    W = N[None]**K[:,None]
    return EkteloMatrix(W)

def WidthKRange(n, widths):
    if type(widths) is int:
        widths = [widths]
    m = sum(n-k+1 for k in widths)
    W = np.zeros((m, n))
    row = 0
    for k in widths:
        for i in range(n-k+1):
            W[row+i, i:i+k] = 1.0
        row += n - k + 1
    return EkteloMatrix(W)

def DimKMarginals(domain, dims):
    if type(dims) is int:
        dims = [dims]
    weights = {}
    for key in itertools.product(*[[1,0]]*len(domain)):
        if sum(key) in dims:
            weights[key] = 1.0
    return Marginals.frombinary(domain, weights)

def Range2D(n):
    return Kronecker([AllRange(n), AllRange(n)])

def Prefix2D(n):
    return Kronecker([Prefix(n), Prefix(n)]) 

def sum_kron_canonical(WtW):
    if isinstance(WtW, Kronecker):
        return Sum([1.0 * WtW])
    elif isinstance(WtW, Weighted) and isinstance(WtW.base, Kronecker):
        return Sum([WtW])
    elif isinstance(WtW, Sum):
        return Sum([1.0 * X for X in WtW.matrices])
    elif isinstance(WtW, Weighted) and isinstance(WtW.base, Sum):
        c = WtW.weight
        return Sum([c * X for X in WtW.base.matrices])
    else:
        raise ValueError('Input format not recognized')
