from ektelo import matrix, workload
from ektelo.hdmm import error
from functools import reduce
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

class TemplateStrategy:

    def strategy(self):
        pass
  
    def _AtA1(self):
        return self.strategy().gram().pinv().dense_matrix()
 
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
        init = np.random.rand(self._params.size)
        bnds = [(0,None)]*init.size
       
        opts = { 'ftol' : 1e-4 }
        res = optimize.minimize(self._loss_and_grad, init, jac=True, method='L-BFGS-B', bounds=bnds, options=opts)
        self._params = res.x
        
    def restart_optimize(self, W, restarts):
        best_A, best_loss = None, np.inf
        for _ in range(restarts):
            self.optimize(W)
            A = self.strategy()
            loss = error.rootmse(W, A)
            if loss <= best_loss:
                best_loss = loss
                best_A = A
        return best_A, best_loss

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

    def _AtA1(self):
        B = np.reshape(self._params, (self.p,self.n))
        scale = 1.0 + np.sum(B, axis=0)
        R = np.linalg.inv(np.eye(self.p) + B @ B.T) # O(k^3)
        return (np.eye(self.n) - B.T @ R @ B)*scale*scale[:,None]
 
    def _set_workload(self, W):
        self._WtW = W.gram().dense_matrix()
        
    def _loss_and_grad(self, params):
        WtW = self._WtW
        p, n = self.p, self.n

        B = np.reshape(params, (p,n))
        scale = 1.0 + np.sum(B, axis=0)
        try: R = np.linalg.inv(np.eye(p) + B.dot(B.T)) # O(k^3)
        except: return np.inf, np.zeros_like(params)
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

    def _AtA1(self):
        params = np.append(0, self._params)
        B = params[self._imatrix]
        self._pid._params = B.flatten()
        return self._pid._AtA1()

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

        WtW = workload.sum_kron_canonical(W.gram())

        workloads = [[Q.dense_matrix() for Q in K.base.matrices] for K in WtW.matrices]
        weights = [K.weight for K in WtW.matrices]
        
        k = len(workloads)
        d = len(workloads[0])
        C = np.ones((d,k))

        for i in range(d):
            temp = self._templates[i]
            # this won't exploit efficient psuedo inverse of pidentity
            AtA1 = temp._AtA1()
            for j in range(k):
                #W = workload.ExplicitGram(workloads[j][i])
                #temp._set_workload(W)
                #C[i,j] = temp._loss_and_grad(temp._params)[0]
                C[i,j] = np.sum(workloads[j][i] * AtA1)
        for _ in range(10):
            #err = C.prod(axis=0).sum()
            for i in range(d):
                temp = self._templates[i]
                cs = C.prod(axis=0) / C[i]
                What = sum(c*WtWs[i] for c, WtWs in zip(cs, workloads))
                What = workload.ExplicitGram(What)
                temp.optimize(What)
                AtA1 = temp._AtA1()
                for j in range(k):
                    #temp._set_workload(workloads[j][i])
                    #C[i,j] = temp._loss_and_grad(temp._params)[0]
                    C[i,j] = np.sum(workloads[j][i] * AtA1)

class Union(TemplateStrategy):
    def __init__(self, templates):
        # expects workload to be a list of same length as templates
        # workload may contain subworkloads defined over different marginals of the data vector:w
        self._templates = templates
        self._weights = np.ones(len(templates)) / len(templates)
    
    def strategy(self):
        return matrix.VStack([w * T.strategy() for w, T in zip(self._weights, self._templates)])
        
    def optimize(self, W):
        assert isinstance(W, list), 'workload must be a list'
        assert len(W) == len(self._templates), 'length of workload list must match templates'
       
        errors = [] 
        for Ti, Wi in zip(self._templates, W):
            Ti.optimize(Wi)
            errors.append(error.expected_error(Wi, Ti.strategy()))

        weights = (2 * np.array(errors))**(1.0/3.0)
        weights /= weights.sum()
        self._weights = weights 
            

class Marginals(TemplateStrategy):
    def __init__(self, domain):
        self._domain = domain
        d = len(domain)
        self._params = np.random.rand(2**len(domain))
 
        self.gram = workload.MarginalsGram(domain, self._params**2)

    def strategy(self):
        return workload.Marginals(self._domain, self._params)

    def _set_workload(self, W):
        marg = workload.Marginals.approximate(W) 
        d = len(self._domain)
        A = np.arange(2**d)
        weights = marg.weights

        self._dphi = np.array([np.dot(weights**2, self.gram._mult[A|b]) for b in range(2**d)]) 

    def _loss_and_grad(self, params):
        d = len(self._domain)
        A = np.arange(2**d)
        mult = self.gram._mult
        Xmatrix = self.gram._Xmatrix
        dphi = self._dphi
        theta = params

        # TODO: accomodate (eps, delta)-DP
        delta = np.sum(theta)**2
        ddelta = 2*np.sum(theta)
        theta2 = theta**2

        Y, YT = Xmatrix(theta2)
        params = Y.dot(theta2)
        X, XT = Xmatrix(params)
        # hack to make solve_triangular work
        D = sparse.diags(X.dot(np.ones(2**d))==0, dtype=float)
        phi = spsolve_triangular(X+D, theta2, lower=False)
        # Note: we should be multiplying by domain size here if we want total squared error
        ans = np.dot(phi, dphi)
        dXvect = -spsolve_triangular(XT+D, dphi, lower=True)
        # dX = outer(dXvect, phi)
        dparams = np.array([np.dot(dXvect[A&b]*phi, mult[A|b]) for b in range(2**d)])
        dtheta2 = YT.dot(dparams)
        dtheta = 2*theta*dtheta2
        return delta*ans, delta*dtheta + ddelta*ans


class YuanConvex(TemplateStrategy):
    def optimize(self, W):
        V = W.gram().dense_matrix()

        accuracy = 1e-10
        max_iter_ls = 50
        max_iter_cg = 5
        theta = 1e-3
        
        beta = 0.5
        sigma = 1e-4
        n = V.shape[0]
        I = np.eye(n)
        X = I
        max_iter = 100
        V = V + theta*np.mean(np.diag(V))*I
        
        iX = I
        G = -V
        fcurr = np.sum((V*iX)**2)
        history = []

        for iter in range(1, max_iter+1):
            if iter == 1:
                D = -G
                np.fill_diagonal(D,0)
                j = 0
            else:
                D = np.zeros((n,n))
                Hx = lambda S: -iX.dot(S).dot(G) - G.dot(S).dot(iX)
                np.fill_diagonal(D, 0)
                R = -G - Hx(D)
                np.fill_diagonal(R, 0)
                P = R;
                rsold = np.sum(R**2)
                for j in range(1, max_iter_cg+1):
                    Hp = Hx(P)
                    alpha = rsold / np.sum(P * Hp)
                    D += alpha*P
                    np.fill_diagonal(D, 0)
                    R -= alpha*Hp
                    np.fill_diagonal(R, 0)
                    rsnew = np.sum(R**2)
                    if np.sqrt(rsnew) < 1e-8:
                        break
                    P = R + rsnew / rsold * P
                    rsold = rsnew

            delta = np.sum(D * G)
            X_old = X
            flast = fcurr
            history.append(fcurr)
            
            for i in range(1, max_iter_ls+1):
                alpha = beta**(i-1)
                X = X_old + alpha*D
                iX = np.linalg.inv(X)
                try:
                    A = np.linalg.cholesky(X)
                except:
                    continue
                G = -iX.dot(V).dot(iX)
                fcurr = np.sum(V * iX)
                if fcurr <= flast + alpha*sigma*delta:
                    break

            #print(fcurr)

            if i==max_iter_ls:
                X = X_old
                fcurr = flast
                break
            if np.abs((flast - fcurr) / flast) < accuracy:
                break

        self.ans = np.linalg.cholesky(X)

    def strategy(self):
        return matrix.EkteloMatrix(self.ans)
 
def KronYuan(ns):
    return Kronecker([YuanConvex() for _ in ns])

def KronPIdentity(ps, ns):
    """
    Builds a template strategy of the form A1 x ... x Ad where each Ai is a PIdentity template
    :param ps: the number of p queries in each dimension
    :param ns: the domain size of each dimension
    """
    return Kronecker([PIdentity(p, n) for p,n in zip(ps, ns)])

def UnionKron(ps, ns):
    """
    Builds a template strategy that is a union of Kronecker products, where each
    kron product is a PIdentity strategy

    :param ps: a table of p values of size k x d where k is number of strategies in union and d in number of dimensions
    :param ns: the domain size of each dimension (length d tuple)
    """
    return Union([KronPIdentity(p, ns) for p in ps])

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
