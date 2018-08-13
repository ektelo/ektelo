import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsqr
from functools import reduce


def diag_like(mat, data, diags, m, n, format=None):
    diag = sparse.spdiags(data, diags, m, n, format)

    if EkteloMatrix in type(mat).mro():
        return EkteloMatrix(diag)
    else:
        return diag
    
def _any_sparse(matrices):
    return any(sparse.issparse(Q.matrix) for Q in matrices)

class EkteloMatrix(LinearOperator):
    """
    An EkteloMatrix is a linear transformation that can compute matrix-vector products 
    """
    # must implement: _matmat, _transpose, matrix
    # can  implement: gram, sensitivity, sum, dense_matrix, sparse_matrix, __abs__, __lstsqr__

    def __init__(self, matrix):
        """ Instantiate an EkteloMatrix from an explicitly represented backing matrix
        
        :param matrix: a 2d numpy array or a scipy sparse matrix
        """
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape
    
    def _transpose(self):
        return EkteloMatrix(self.matrix.T)
    
    def _matmat(self, V):
        """
        Matrix multiplication of a m x n matrix Q
        
        :param V: a n x p numpy array
        :return Q*V: a m x p numpy aray
        """
        return self.matrix @ V

    def gram(self):
        """ 
        Compute the Gram matrix of the given matrix.
        For a matrix Q, the gram matrix is defined as Q^T Q
        """
        return self.T @ self # works for subclasses too
   
    def sensitivity(self):
        # note: this works because np.abs calls self.__abs__
        return np.max(np.abs(self).sum(axis=1))
 
    def sum(self, axis=None, dtype=None, out=None):
        # GDB: I dropped my pass-through implementation in here because
        # there were problem with your implementations (see below).
        #return self.matrix.sum(axis, dtype, out)
        
        # RM: is it worth it to conform to the numpy api and support dtype/out? What does this buy us?
        
        # this implementation works for all subclasses too (as long as they define _matmat and _transpose)
        if axis == 0:
            return self.dot(np.ones(self.shape[1]))
        ans = self.T.dot(np.ones(self.shape[0]))  
        return ans if axis == 1 else np.sum(ans)
    
    # deprecate this if possible, only works with sparse matrix backing
    # should call dense_matrix instead
    def toarray(self):
        return self.dense_matrix()

    def tocsr(self):
        return sparse.csr_matrix(self.matrix)

    def _adjoint(self):
        return self._transpose()

    def __mul__(self, other):
        # GDB: I had to bring over my implementation because there are places
        # in the plans where we use the "*" operator.
        
        # if other is a numpy array, simply call dot
        # if other is an EkteloMatrix, otherwise perform multiplication
        if np.isscalar(other):
            return Weighted(self, other)
        if type(other) == np.ndarray:
            return self.dot(other)
        if isinstance(other, EkteloMatrix):
            # note: this expects both matrix types to be compatible (e.g., sparse and sparse)
            # todo: make it work for different backing representations
            return EkteloMatrix(self.matrix @ other.matrix)
        # todo: deprecate this if possible (shouldn't be allowed to multiply with scipy)
        if sparse.compressed._cs_matrix in type(other).mro():
            return EkteloMatrix(self.matrix * other)
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloMatrix' % type(other))
            
    def __rmul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        return NotImplemented
    
    def dense_matrix(self):
        """
        return the dense representation of this matrix, as a 2D numpy array
        """
        if sparse.issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix
    
    def sparse_matrix(self):
        """
        return the sparse representation of this matrix, as a scipy matrix
        """
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.matrix)
    
    @property
    def ndim(self):
        # todo: deprecate if possible
        return 2
    
    def __abs__(self):
        return EkteloMatrix(self.matrix.__abs__())
    
    # RM: consider deprecating this
    def __lstsqr__(self, y):
        """
        solve a least squares problem with this matrix.
        For a matrix Q and a vector y, the least square solution is the vector x
        such that || Qx - y ||_2 is minimized
        """
        # works for subclasses too
        return lsqr(self, y)[0]

class Identity(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n,n)
        self.dtype = dtype
   
    def _matmat(self, V):
        return V
 
    def _transpose(self):
        return self

    @property
    def matrix(self):
        return sparse.eye(self.n, dtype=self.dtype)
    
    def __mul__(self, other):
        assert other.shape[0] == self.n, 'dimension mismatch'
        return other

    def __abs__(self):  
        return self

    def __lstsqr__(self, v):
        return v

class Ones(EkteloMatrix):
    """ A m x n matrix of all ones """
    def __init__(self, m, n, dtype=np.float64):
        self.m = m
        self.n = n
        self.shape = (m,n)
        self.dtype = dtype
        
    def _matmat(self, V):
        ans = V.sum(axis=0, keepdims=True)
        return np.repeat(ans, self.m, axis=0)
    
    def _transpose(self):
        return Ones(self.n, self.m, self.dtype)
    
    def gram(self):
        return self.m * Ones(self.n, self.n, self.dtype)
        
    @property
    def matrix(self):
        return np.ones(self.shape, dtype=self.dtype)
    
    def __abs__(self):
        return self
    
class Weighted(EkteloMatrix):
    """ Class for multiplication by a constant """
    def __init__(self, base, weight):
        self.base = base
        self.weight = weight
        self.shape = base.shape
        self.dtype = base.dtype
    
    def _matmat(self, V):
        return self.weight * self.base.dot(V)
    
    def _transpose(self):
        return Weighted(self.base.T, self.weight)
    
    def gram(self):
        return Weighted(self.base.gram(), self.weight**2)
    
    def __abs__(self):
        return Weighted(self.base.__abs__(), np.abs(self.weight))
    
    @property
    def matrix(self):
        return self.weight * self.base.matrix

class Sum(EkteloMatrix):
    """ Class for the Sum of matrices """
    def __init__(self, matrices):
        # all must have same shape
        self.matrices = matrices
        self.shape = matrices[0].shape
        self.dtype = matrices[0].dtype # RM: what to do if dtypes differ?

    def _matmat(self, V):
        return sum(Q.dot(V) for Q in self.matrices)

    def _transpose(self):
        return Sum([Q.T for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return sum(Q.sparse_matrix() for Q in self.matrices)
        return sum(Q.dense_matrix() for Q in self.matrices)

class VStack(EkteloMatrix):
    def __init__(self, matrices):
        m = sum(Q.shape[0] for Q in matrices)
        n = matrices[0].shape[1]
        assert all(Q.shape[1] == n for Q in matrices), 'dimension mismatch'
        self.shape = (m,n)
        self.matrices = matrices
        self.dtype = matrices[0].dtype # what if dtypes differ?
    
    def _matmat(self, V):
        return np.vstack([Q.dot(V) for Q in self.matrices])

    def _transpose(self):
        return HStack([Q.T for Q in self.matrices])
    
    def __mul__(self, other):
        if isinstance(other,EkteloMatrix):
            return VStack([Q @ other for Q in self.matrices]) # should use others rmul though
        return EkteloMatrix.__mul__(self, other)

    def gram(self):
        return Sum([Q.gram() for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return self.sparse_matrix()
        return self.dense_matrix()

    def dense_matrix(self):
        return np.vstack([Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return sparse.vstack([Q.sparse_matrix() for Q in self.matrices])

    def __abs__(self):
        return VStack([Q.__abs__() for Q in self.matrices])

class HStack(EkteloMatrix):
    def __init__(self, matrices):
        # all matrices must have same number of rows
        cols = [Q.shape[1] for Q in matrices]
        m = matrices[0].shape[0]
        n = sum(cols)
        assert all(Q.shape[0] == m for Q in matrices), 'dimension mismatch'
        self.shape = (m,n)
        self.matrices = matrices
        self.dtype = matrices[0].dtype
        self.split = np.cumsum(cols)[::-1]

    def _matmat(self, V):
        vs = np.split(V, self.split)
        return sum([Q.dot(z) for Q, z in zip(self.matrices, vs)])
    
    def _transpose(self):
        return VStack([Q.T for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return self.sparse_matrix()
        return self.dense_matrix()
    
    def dense_matrix(self):
        return np.hstack([Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return sparse.hstack([Q.sparse_matrix() for Q in self.matrices])
    
    def __mul__(self, other):
        if isinstance(other, VStack):
            # and shapes match...
            return Sum([A @ B for A,B in zip(self.matrices, other.matrices)])
        return EkteloMatrix.__mul__(self, other)

    def __abs__(self):
        return HStack([Q.__abs__() for Q in self.matrices])

class Kronecker(EkteloMatrix):
    def __init__(self, matrices):
        self.matrices = matrices
        self.shape = tuple(np.prod([Q.shape for Q in matrices], axis=0))
        self.dtype = matrices[0].dtype

    def _matvec(self, v):
        size = self.shape[1]
        X = v
        for Q in self.matrices[::-1]:
            m, n = Q.shape
            X = Q @ X.reshape(size//n, n).T
            size = size * m // n
        return X.flatten()

    def _transpose(self):
        return Kronecker([Q.T for Q in self.matrices]) 
   
    def gram(self):
        return Kronecker([Q.gram() for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return self.sparse_matrix()
        return self.dense_matrix()
 
    def dense_matrix(self):
        return reduce(np.kron, [Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return reduce(sparse.kron, [Q.sparse_matrix() for Q in self.matrices])
  
    def sensitivity(self):
        return np.prod([Q.sensitivity() for Q in self.matrices])
    
    def __mul__(self, other):
        # perform the multiplication in the implicit representation if possible
        if isinstance(other, Kronecker):
            return Kronecker([A @ B for A,B in zip(self.matrices, other.matrices)])
        return EkteloMatrix.__mul__(self, other)
 
    def __abs__(self):
        return Kronecker([Q.__abs__() for Q in self.matrices]) 

    def __lstsqr__(self, v):
        pass

if __name__ == '__main__':
    I = Identity(5)
    T = Ones(1,5)
    A = VStack([I,T])
    G = A.gram()
    print(G.matrix)
    B = Kronecker([A,A])
    print(B.shape)
    print(B.gram(), B.T * B, B.T @ B)
    print(B.dense_matrix().sum())
    X = EkteloMatrix(np.random.rand(25,36))
    Y = X * B
    print(Y.shape, type(B), type(X), type(Y))
    Z = B * X
    print(Z.shape, type(Z))
