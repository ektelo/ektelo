import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsqr


def diag_like(mat, data, diags, m, n, format=None):
    diag = sparse.spdiags(data, diags, m, n, format)

    if EkteloMatrix in type(mat).mro():
        return EkteloMatrix(diag)
    else:
        return diag

class EkteloMatrix(LinearOperator):
    # must implement: _matmat, transpose
    # can  implement: gram, sensitivity, sum, dense_matrix, spares_matrix, __abs__, __lstsqr__

    def __init__(self, matrix):
        # matrix may be one of:
        #  1) 2D numpy array
        #  2) scipy sparse matrix
        #  3) scipy linear operator * (but __abs__ isn't supported here)
        # note: dtype may also vary (e.g., 32 bit or 64 bit float) 
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape
        self.ndim = 2
    
    def _transpose(self):
        return EkteloMatrix(self.matrix.T)
    
    def _matmat(self, V):
        return self.matrix @ V

    def gram(self):
        return EkteloMatrix(self.matrix.T @ self.matrix)
   
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
    def toarray(self):
        return self.matrix.toarray()

    def _adjoint(self):
        return self.transpose()

    def __mul__(self, other):
        # implement carefully -- what to do if backing matrix types differ?
        # GDB: I had to bring over my implementation because there are places
        # in the plans where we use the "*" operator.
        
        # if other is a numpy array, simply call dot
        # if other is an EkteloMatrix, otherwise perform multiplication
        if type(other) == np.ndarray:
            return self.dot(other)
        if type(other) == EkteloMatrix:
            # note: this expects both matrix types to be compatible (e.g., sparse and sparse)
            # todo: make it work for different backing representations
            return EkteloMatrix(self.matrix @ other.matrix)
        # todo: deprecate this if possible (shouldn't be allowed to do this)
        if sparse.compressed._cs_matrix in type(other).mro():
            return EkteloMatrix(self.matrix * other)
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloMatrix' % type(other))
    
    # should this return a numpy array or an ektelo matrix with numpy array backing?
    def dense_matrix(self):
        return self.dot(np.eye(self.shape[0]))
    
    # should this return a sparse matrix or an ektelo matrix with sparse matrix backing?
    def sparse_matrix(self):
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.dense_matrix()) 
    
    def __abs__(self):
        # note: note implemented if self.matrix is a linear operator
        # what should we do in this case?  Fail or convert to a dense backing representation?
        # (perhaps throwing a warning)
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

    def gram(self):
        return self

    def dense_matrix(self):
        return np.eye(self.n, self.dtype)

    def sparse_matrix(self):
        return sparse.eye(self.n, self.dtype)

    def __abs__(self):  
        return self

    def __lstsqr__(self, v):
        return v

class Total(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        EkteloMatrix.__init__(self, np.ones((1,n), dtype))
    
    def __lstsqr__(self, v):
        return self.T.dot(v) / self.n

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
        # for a matrix A = [A1; A2; ...] and a matrix B, we have AB = [A1 B; A2 B; ...]
        return VStack([Q @ other for Q in self.matrices])

    def gram(self):
        return Sum([Q.gram() for Q in self.matrices])

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
        self.split = np.cumsum(cols)[::-1]

    def matvec(self, v):
        vs = np.split(v, self.split)
        return sum([Q.dot(z) for Q, z in zip(self.matrices, vs)])
    
    def _transpose(self):
        return VStack([Q.T for Q in self.matrices])

    def __abs__(self):
        return HStack([Q.__abs__() for Q in self.matrices])

def Kronecker(EkteloMatrix):
    def __init__(self, matrices):
        self.matrices = matrices
        self.shape = tuple(np.prod([Q.shape for Q in matrices]))
        self.dtype = matrices[0].dtype

    def _matvec(self, v):
        size = self.shape[1]
        X = v
        for Q in self.matrices[::-1]:
            m, n = Q.shape
            X = Q @ X.reshape(size//n, n).T
            size = size * m // n
        return X.flatten()

    def _transpose(self, v):
        return Kronecker([Q.T for Q in self.matrices]) 
   
    def gram(self):
        return Kronecker([Q.gram() for Q in self.matrices])
 
    def dense_matrix(self):
        return reduce(np.kron, [Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return reduce(sparse.kron, [Q.sparse_matrix() for Q in self.matrices])
  
    def sensitivity(self):
        return np.prod([Q.sensitivity() for Q in self.matrices])
 
    def __abs__(self):
        return Kronecker([Q.__abs__() for Q in self.matrices]) 

    def __lstsqr__(self, v):
        pass

if __name__ == '__main__':
    A = EkteloMatrix(aslinearoperator(np.eye(5)))
    A = EkteloMatrix(np.eye(5))
    x = np.random.rand(5)
    # multiple ways to do matrix-vector product (all inherited from LinearOperator)
    print(A.dot(x), A*x, A@x)
