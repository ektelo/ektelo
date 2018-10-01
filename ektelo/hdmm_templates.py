from ektelo.matrix import *


class TemplateStrategy(EkteloMatrix):

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
        self.set_params(res.x)
         
class Default(TemplateStrategy):
    def __init__(self, m, n):
        self._params = np.random.rand(m*n)
        self.shape = (m, n)

    def strategy(self):
        A = self._params.reshape(self.shape)
        return EkteloMatrix(A)

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
        return EkteloMatrix(A / A.sum(axis=0))
 
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

    def strategy(self):
        params = np.append(0, params)
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

    def optimize(self):
        pass

class Kronecker(TemplateStrategy):
    def __init__(self, strategies):
        pass


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
