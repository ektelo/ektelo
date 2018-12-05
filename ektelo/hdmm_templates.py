from ektelo import matrix, workload
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
import numpy as np
from functools import reduce
import itertools

class TemplateStrategy:

    def strategy(self):
        pass
   
    def _loss_and_grad(self, params):
        pass

    def _set_workload(self, W):
        self._workload = W
        self._gram = W.gram()

    def optimize(self, W):
        """
        Optimize strategy for given workload 
        :param W: the workload, may be a n x n numpy array for WtW or a workload object
        """
        self._set_workload(W)
        init = self._params
        bnds = [(0,None)]*init.size
       
        opts = { 'ftol' : 1e-4 }
        res = optimize.minimize(self._loss_and_grad, init, jac=True, method='L-BFGS-B', bounds=bnds, options=opts)
        self._params = res.x
         
class Default(TemplateStrategy):
    def __init__(self, m, n):
        self._params = np.random.rand(m*n)
        self.shape = (m, n)

    def strategy(self):
        A = self._params.reshape(self.shape)
        return matrix.EkteloMatrix(A)

    def _set_workload(self, W):
        self._WtW = W.gram().dense_matrix()
    
    def _loss_and_grad(self, params):
        WtW = self._WtW
        A = params.reshape(self.shape)
        sums = np.sum(np.abs(A), axis=0)
        col = np.argmax(sums)
        F = sums[col]**2
        # note: F is not differentiable, but we can take subgradients
        dF = np.zeros_like(A)
        dF[:,col] = np.sign(A[:,col])*2*sums[col]
        AtA = A.T.dot(A)
        AtA1 = np.linalg.pinv(AtA)
        M = WtW.dot(AtA1)
        G = np.trace(M)
        dX = -AtA1.dot(M)
        dG = 2*A.dot(dX)
        dA = dF*G + F*dG
        return F*G, dA.flatten()

class PIdentity(TemplateStrategy):
    """
    A PIdentity strategy is a strategy of the form (I + B) D where D is a diagonal scaling matrix
    that depends on B and ensures uniform column norm.  B is a p x n matrix of free parameters.
    """
    def __init__(self, p, n):
        """
        Initialize a PIdentity strategy
        :param p: the number of non-identity queries
        :param n: the domain size
        """
        self._params = np.random.rand(p*n)
        self.p = p
        self.n = n

    def strategy(self):
        B = sparse.csr_matrix(self._params.reshape(self.p, self.n))
        I = sparse.eye(self.n, format='csr')
        A = sparse.vstack([I, B], format='csr')
        return matrix.EkteloMatrix(A / A.sum(axis=0))
 
    def _set_workload(self, W):
        self._WtW = W.gram().dense_matrix()
        
    def _loss_and_grad(self, params):
        WtW = self._WtW
        p, n = self.p, self.n

        B = np.reshape(params, (p,n))
        scale = 1.0 + np.sum(B, axis=0)
        R = np.linalg.inv(np.eye(p) + B.dot(B.T)) # O(k^3)
        C = WtW * scale * scale[:,None] # O(n^2)
        M1 = R.dot(B) # O(n k^2)
        M2 = M1.dot(C) # O(n^2 k)
        M3 = B.T.dot(M2) # O(n^2 k)
        M4 = B.T.dot(M2.dot(M1.T)).dot(B) # O(n^2 k)

        Z = -(C - M3 - M3.T + M4) * scale * scale[:,None] # O(n^2)

        Y1 = 2*np.diag(Z) / scale # O(n)
        Y2 = 2*(B/scale).dot(Z) # O(n^2 k)
        g = Y1 + (B*Y2).sum(axis=0) # O(n k)

        loss = np.trace(C) - np.trace(M3)
        grad = (Y2*scale - g) / scale**2
        return loss, grad.flatten()

class AugmentedIdentity(TemplateStrategy):
    """
    An AugmentedIdentity strategy is like a PIdentity strategy with additional structure imposed.
    The template is defiend by a p x n matrix of non-negative integers P.  Each unique nonzero entry
    of this matrix P refers to a free parameter that can be optimized.  An entry that is 0 in P is
    a structural zero in the strategy.  
    Example 1:
    A PIdentity strategy can be represented as an AugmentedIdentity strategy with 
    P = np.arange(1, p*n+1).reshape(p, n)
    
    Example 2:
    A strategy of the form w*T + I can be represented as an AugmentedIdentity strategy with
    P = np.ones((1, n), dtype=int)
    """
    def __init__(self, imatrix):
        self._imatrix = imatrix
        p, n = imatrix.shape
        num = imatrix.max()
        self._params = np.random.rand(num)
        self._pid = PIdentity(p, n)

    def _set_workload(self, W):
        self._pid._set_workload(W)

    def strategy(self):
        params = np.append(0, self._params)
        B = params[self._imatrix]
        self._pid._params = B.flatten()
        return self._pid.strategy()

    def _loss_and_grad(self, params):
        params = np.append(0, params)
        B = params[self._imatrix]
        obj, grad = self._pid._loss_and_grad(B.flatten())
        grad2 = np.bincount(self._imatrix.flatten(), grad)[1:]
        return obj, grad2
         

class Static(TemplateStrategy):
    def __init__(self, strategy):
        self._strategy = strategy

    def strategy(self):
        return self._strategy

    def optimize(self, W):
        pass

class Kronecker(TemplateStrategy):
    def __init__(self, templates):
        self._templates = templates

    def strategy(self):
        return matrix.Kronecker([T.strategy() for T in self._templates])
        
    def optimize(self, W):
        if isinstance(W, matrix.Kronecker):
            for subA, subW in zip(self._templates, W.matrices):
                subA.optimize(subW)
            return
        assert isinstance(W, matrix.VStack) and isinstance(W.matrices[0], matrix.Kronecker)
        
        workloads = [K.matrices for K in W.matrices] # a k x d table of workloads
        k = len(workloads)
        d = len(workloads[0])
        C = np.ones((d,k))

        for i in range(d):
            temp = self._templates[i]
            for j in range(k):
                temp._set_workload(workloads[j][i])
                C[j][i] = temp._loss_and_grad(temp._params)[0]
        for _ in range(10):
            #err = C.prod(axis=0).sum()
            for i in range(d):
                temp = self._templates[i]
                cs = np.sqrt(C.prod(axis=0) / C[i])
                What = matrix.VStack([c*Ws[i] for c, Ws in zip(cs, workloads)])
                temp.optimize(What)
                for j in range(k):
                    temp._set_workload(workloads[j][i])
                    C[j][i] = temp._loss_and_grad(temp._params)[0]

class Marginals(TemplateStrategy):
    def __init__(self, domain):
        self._domain = domain
        d = len(domain)
        self._params = np.random.rand(2**len(domain))
        mult = np.ones(2**d)
        for i in range(2**d):
            for k in range(d):
                if not (i & (2**k)):
                    mult[i] *= domain[k]
        self._mult = mult

    def strategy(self):
        dom = self._domain
        keys = itertools.product(*[[0,1]]*len(dom))
        weights = dict(zip(keys, np.sqrt(self._params)))
        return workload.Marginals(dom, weights) 

    def _set_workload(self, W):
        marg = marginals_approx(W)
        d = len(self._domain)
        A = np.arange(2**d)

        weights = np.zeros(2**d)
        for i in range(2**d):
            key = tuple([int(bool(2**k & i)) for k in range(d)])
            weights[i] = marg.weights[key]

        self._dphi = np.array([np.dot(weights**2, self._mult[A|b]) for b in range(2**d)]) 

    def _Xmatrix(self,vect):
        # the matrix X such that M(u) M(v) = M(X(u) v)
        d = len(self._domain)
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
            if values[start+step-1] == 0:
                values[start+step-1] = 1.0 # hack to make solve triangular work
            cols[start:start+step] = b
            rows[start:start+step] = uniq
            start += step
        X = sparse.csr_matrix((values, (rows, cols)), (2**d, 2**d))
        XT = sparse.csr_matrix((values, (cols, rows)), (2**d, 2**d))
        return X, XT

    def _loss_and_grad(self, params):
        d = len(self._domain)
        A = np.arange(2**d)
        mult = self._mult
        dphi = self._dphi
        theta = params

        delta = np.sum(theta)**2
        ddelta = 2*np.sum(theta)
        theta2 = theta**2
        Y, YT = self._Xmatrix(theta2)
        params = Y.dot(theta2)
        X, XT = self._Xmatrix(params)
        phi = spsolve_triangular(X, theta2, lower=False)
        # Note: we should be multiplying by domain size here if we want total squared error
        ans = np.dot(phi, dphi)
        dXvect = -spsolve_triangular(XT, dphi, lower=True)
        # dX = outer(dXvect, phi)
        dparams = np.array([np.dot(dXvect[A&b]*phi, mult[A|b]) for b in range(2**d)])
        dtheta2 = YT.dot(dparams)
        dtheta = 2*theta*dtheta2
        return delta*ans, delta*dtheta + ddelta*ans

def KronPIdentity(ps, ns):
    """
    Builds a template strategy of the form A1 x ... x Ad where each Ai is a PIdentity template
    :param ps: the number of p queries in each dimension
    :param ns: the domain size of each dimension
    """
    return Kronecker([PIdentity(p, n) for p,n in zip(ps, ns)])

def RangeTemplate(n, start=32, branch=4, shared=False):
    """
    Builds a template strategy for range queries with queries that have structural zeros 
    everywhere except at indices at [i, i+w) where w is the width of the query and ranges from
    start to n in powers of branch and i is a multiple of w/2.
    :param n: the domain size
    :param start: the width of the smallest query
    :param branch: the width multiplying factor for larger queries
    :param shared: flag to determine if parameters should be shared for queries of the same width
    Example:
    RangeTemplate(16, start=8, branch=2) builds a strategy template with four augmented queries that have structural zeros everywhere except in the intervals indicated below:
    1. [0,8)
    2. [4,12)
    3. [8,16)
    4. [0,16)
    """
    rows = []
    width = start
    idx = 1
    while width <= n:
        for i in range(0, n-width//2, width//2):
            row = np.zeros(n, dtype=int)
            row[i:i+width] = np.arange(width) + idx
            if not shared: idx += width
            rows.append(row)
        if shared: idx += width
        width *= branch
    return AugmentedIdentity(np.vstack(rows))

def IdTotal(n):
    """ Build a single-parameter template strategy of the form w*Total + Identity """
    P = np.ones((1,n), dtype=int)
    return AugmentedIdentity(P)

def Identity(n):
    """ Builds a template strategy that is always Identity """
    return Static(np.eye(n))

def Total(n):
    """ Builds a template strategy that is always Total """
    return Static(np.ones((1,n)))

def marginals_approx(W):
    """
    Given a Union-of-Kron workload, find a Marginals workload that approximates it.
    
    The guarantee is that for all marginals strategies A, Error(W, A) = Error(M, A) where
    M is the returned marginals approximation of W.
    The other guarantee is that this function is idempotent: approx(approx(W)) = approx(W)
    """
    if isinstance(W, matrix.Kronecker):
        W = matrix.VStack([W])
    assert isinstance(W, matrix.VStack) and isinstance(W.matrices[0], matrix.Kronecker)
    dom = tuple(Wi.shape[1] for Wi in W.matrices[0].matrices)
    weights = np.zeros(2**len(dom))
    for sub in W.matrices:
        tmp = []
        for n, piece in zip(dom, sub.matrices):
            X = piece.gram().dense_matrix()
            b = float(X.sum() - X.trace()) / (n * (n-1))
            a = float(X.trace()) / n - b
            tmp.append(np.array([b,a]))
        weights += reduce(np.kron, tmp)
    keys = itertools.product(*[[0,1]]*len(dom))
    weights = dict(zip(keys, np.sqrt(weights)))
    return workload.Marginals(dom, weights) 
