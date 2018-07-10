from matrix import *
import collections

# workloads

class IdentityTotal(VStack):
    def __init__(self, n):
        sub = [Identity(n), Total(n)]
        VStack.__init__(self, sub)

class Prefix(EkteloMatrix):
    def __init__(self, n):
        self.n = n
        self.shape = (n,n)

    def matmat(self, V):
        return np.cumsum(V, axis=0)

    def transpose(self):
        return Suffix(self.n) 

    def dense_matrix(self):
        return np.tril(np.ones((self.n, self.n)))

    def __abs__(self);
        return self

class Suffix(EkteloMatrix):
    def __init__(self, n):
        self.n = n
        self.shape = (n,n)

    def matmat(self, V):
        return np.cumsum(V[::-1], axis=0)[::-1]
    
    def transpose(self):
        return Prefix(self.n)
    
    def dense_matrix(self):
        return np.triu(np.ones((self.n, self.n)))

    def __abs__(self):
        return self

class AllRange(EkteloMatrix):
    def __init__(self, n):
        self.shape = ((n*(n+1) // 2), n)
        
    def matmat(self):
        pass

    def gram(self):
        r = np.arange(self.n) + 1
        X = np.outer(r, r[::-1])
        reutrn EkteloMatrix(np.minimum(X, X.T))

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
    return Kron([AllRange(n), AllRange(n)])

def Prefix2D(n):
    return Kron([Prefix(n), Prefix(n)]) 
