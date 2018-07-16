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
    An EkteloMatrix is a linear transformation 
    """
    # must implement: _matmat, _transpose, matrix
    # can  implement: gram, sensitivity, sum, dense_matrix, sparse_matrix, __abs__, __lstsqr__

    def __init__(self, matrix):
        # matrix may be one of:
        #  1) 2D numpy array
        #  2) scipy sparse matrix
        # note: dtype may also vary (e.g., 32 bit or 64 bit float) 
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape
    
    def _transpose(self):
        return EkteloMatrix(self.matrix.T)
    
    def _matmat(self, V):
        return self.matrix @ V

    def gram(self):
        return self.T @ self # works for subclasses too
        # EkteloMatrix(self.matrix.T @ self.matrix)
   
    def sensitivity(self):
        # note: this works because np.abs calls self.__abs__
        return np.max(np.abs(self).sum(axis=1))
 
    def sum(self, axis=None, dtype=None, out=None):
        # GDB: I dropped my pass-through implementation in here because
        # there were problem with your implementations (see below).
        #return self.matrix.sum(axis, dtype, out)

        if axis == 0:
            return self.dot(np.ones(self.shape[1]))
        ans = self.T.dot(np.ones(self.shape[0]))  
        return ans if axis == 1 else np.sum(ans)
    
    # deprecate this if possible, only works with sparse matrix backing
    # should call dense_matrix instead
    def toarray(self):
        return self.matrix.toarray()

    def _adjoint(self):
        return self._transpose()

    def __mul__(self, other):
        # implement carefully -- what to do if backing matrix types differ?
        # GDB: I had to bring over my implementation because there are places
        # in the plans where we use the "*" operator.
        
        # if other is a numpy array, simply call dot
        # if other is an EkteloMatrix, otherwise perform multiplication
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
    
    # should this return a numpy array or an ektelo matrix with numpy array backing?
    def dense_matrix(self):
        if sparse.issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix
    
    # should this return a sparse matrix or an ektelo matrix with sparse matrix backing?
    def sparse_matrix(self):
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.matrix)
    
    @property
    def ndim(self):
        # todo: deprecate if possible
        return 2
    
    def __abs__(self):
        return EkteloMatrix(self.matrix.__abs__())
    
    def __lstsqr__(self, v):
        # works for subclasses too
        return lsqr(self, v)[0]

class Identity(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        # GDB: this and other subclasses probably need to implement
        # things like shape, dtype, ndim
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
        return other

    def __abs__(self):  
        return self

    def __lstsqr__(self, v):
        return v

class Ones(EkteloMatrix):
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
        
    @property
    def matrix(self):
        return np.ones(self.shape, dtype=self.dtype)

class Sum(EkteloMatrix):
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
    
# For a VStack object Q we can do the following:
# A * Q where A is a HStack (returns a Sum)
# Q * A where A is an EkteloMatrix
# Q * A where A is a HStack (returns a VStack of HStacks)
# note: A * Q where A is an EkteloMatrix requires either 
#         (1) converting Q to it's explicit representation or 
#         (2) splitting A so that it is a HStack object
        
# What if A is a Kronecker product that has the right size?
# should we support these operations, defaulting to the explicit representations when necessary?
# or should we just fail

class VStack(EkteloMatrix):
    def __init__(self, matrices):
        # GDB: this needs to implement ndim as a member in order to support "*"
        # all matrices must have same number of columns
        self.matrices = matrices
        m = sum(Q.shape[0] for Q in matrices)
        n = matrices[0].shape[1]
        self.shape = (m,n)
        self.dtype = matrices[0].dtype
    
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
        self.matrices = matrices
        cols = [Q.shape[1] for Q in matrices]
        m = matrices[0].shape[0]
        n = sum(cols)
        self.shape = (m,n)
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
