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

class RandomRange(EkteloMatrix):
    def __init__(self, shape_list, domain_shape, size, seed=9001):
        self.shape_list = list(shape_list)
        self.seed = seed
        self.size = size

        prng = np.random.RandomState(seed)
    
class Wavelet(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        # todo: check n = 2^k
        # for now default to the standard EkteloMatrix operations
        # in the future implement the fast (linear time) matrix-vector product algorithm
        self.n = n
        self.shape = (n,n)
        self.dtype = dtype
        
    def _matmat(self, V):
        return NotImplemented # todo
    
    def _transpose(self):
        return NotImplemented # todo
    
    @property
    def matrix(self):
        return _wavelet_sparse(self.n).astype(self.dtype)
    
class Hierarchical(EkteloMatrix):
    def __init__(self, n, branch=2, dtype=np.float64):
        self.n = n
        self.branch = branch
        self.dtype = dtype
        self.shape = (_hierarchical_rows(n, branch), n)
        
    def _matvec(self, x):
        m = x.shape[0]
        b = self.branch
        ans = [x]
        while m > 1:
            r = m % b
            y = sum(x[i::b] for i in range(r,b))
            if r > 0:
                y = sum(x[i::b] for i in range(r)) + np.append(y,0)
            x = y
            m = x.shape[0]
            ans.append(x)
        return np.concatenate(ans[::-1])
    
    def _transpose(self):
        return NotImplemented
    
    @property
    def matrix(self):
        H = _hierarchical_sparse(self.n, self.branch)
        widths = np.array(H.sum(axis=1)).flatten()
        perm = np.argsort(-widths)
        return H[perm]

class PIdentity(EkteloMatrix):
    def __init__(self, theta):
        self.theta = theta
        p, n = theta.shape
        self.shape = (p+n, n)
        Q = np.vstack([np.eye(n), theta])
        Q /= np.abs(Q).sum(axis=0)
        EkteloMatrix.__init__(self, Q)

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

def _wavelet_sparse(n):
    '''
    Returns a sparse (csr_matrix) wavelet matrix of size n = 2^k
    '''
    if n == 1:
        return sparse.identity(1, format='csr')
    m, r = divmod(n, 2)
    assert r == 0, 'n must be power of 2'
    H2 = Wavelet.wavelet_sparse(m)
    I2 = sparse.identity(m, format='csr')
    A = sparse.kron(H2, [1,1])
    B = sparse.kron(I2, [1,-1])
    return sparse.vstack([A,B])

def _hierarchical_rows(n,b):
    if n <= 1: return n
    m, r = divmod(n,b)
    rows0 = _hierarchical_rows(m, b)
    rows1 = _hierarchical_rows(m+1, b) if r>0 else 0
    return 1 + r*rows1 + (b-r)*rows0

def _hierarchical_sparse(n, b):
    '''
    Builds a sparsely represented (csr_matrix) hierarchical matrix
    with n columns and a branching factor of b.  Works even when n
    is not a power of b
    '''
    if n == 0: return sparse.csr_matrix((0,0))
    if n == 1: return sparse.csr_matrix([1.0])

    m, r = divmod(n, b)
    hier0 = _hierarchical_sparse(m, b) 
    hier1 = _hierarchical_sparse(m+1, b) if r>0 else None 
    total = np.ones((1,n))
    sub = sparse.block_diag([hier1]*r + [hier0]*(b-r), format='csr')
    return sparse.vstack([total, sub], format='csr')


