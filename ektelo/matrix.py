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
    # must implement: matvec, transpose
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
        self.ndim = matrix.ndim # GDB: this seems to be necessary for "*"
    
    def transpose(self):
        return EkteloMatrix(self.matrix.T)
    
    def matmat(self, u):
        return self.matrix @ u
    
    # GDB: this is needed for dot
    def matvec( self, x ):
        return self.matrix * x

    def max(self, axis=None, out=None):
        return self.matrix.max(axis, out)

    def gram(self):
        return EkteloMatrix(self.matrix.T @ self.matrix)
   
    def sensitivity(self):
        return np.max(np.abs(self).sum(axis=1))
 
    def sum(self, axis=None, dtype=None, out=None):
        # GDB: I dropped my pass-through implementation in here because
        # there were problem with your implementations (see below).
        return self.matrix.sum(axis, dtype, out)

        if axis == 0:
            return self * np.ones(self.shape[1]) # GDB: this is returning None
        ans = np.ones(self.shape[0]) * self  # GDB: this won't work because np.array does not 
                                             # know how to multiply with EkteloMatrix
        return ans if axis == 1 else np.sum(ans)
    
    def toarray(self):
        return self.matrix.toarray()

    def adjoint(self):
        return self.transpose()

    def __mul__(self, other):
        # implement carefully -- what to do if backing matrix types differ?
        # GDB: I had to bring over my implementation because there are places
        # in the plans where we use the "*" operator.
        if type(other) == EkteloMatrix:
            other = other.matrix

        if sparse.compressed._cs_matrix in type(other).mro():
            return EkteloMatrix(self.matrix * other)
        elif type(other) == np.ndarray:
            return self.matrix * other
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloMatrix' % type(other))
    
    # GDB: I had to remove this; the fact that it does not return
    # a numpy array causes problems with np.abs
    """
    def __array__(self):
        return self.dense_matrix()
    """

    def dense_matrix(self):
        return self * np.eye(self.shape[0])
    
    def sparse_matrix(self):
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.dense_matrix()) 
    
    def __abs__(self):
        # note: note implemented if self.matrix is a linear operator
        return EkteloMatrix(self.matrix.__abs__())
    
    def __lstsqr__(self, v):
        # works for subclasses too
        return lsqr(self, v)[0]

    # GDB: I had to add this because numpy with throw on occasion
    # without it. Actually the docs for LinearOperator do specify 
    # that either this or _matmat must be implemented. 
    def _matvec(self, b):
        raise NotImplementedError( "_matvec" )

def Identity(EkteloMatrix):
    def __init__(self, n):
        # GDB: this and other subclasses probably need to implement
        # things like shape, dtype, ndim
        self.n = n
        self.shape = (n,n)
   
    def matvec(self, v):
        return v
 
    def transpose(self):
        return self

    def gram(self):
        return self

    def dense_matrix(self):
        return np.eye(self.n)

    def sparse_matrix(self):
        return sparse.eye(self.n)

    def __abs__(self):  
        return self

    def __lstsqr__(self, v):
        return v

class Total(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (1, n)
        self.dtype = dtype

    def matvec(self, v):
        return np.array([np.sum(v)])
    
    def rmatvec(self, v):
        return np.ones(self.n) * v[0]
    
    def __abs__(self):
        return self
    
    def __lstsqr__(self, v):
        return self.T.dot(v) / self.n

class Sum(EkteloMatrix):
    def __init__(self, matrices):
        self.matrices = matrices
        self.shape = matrices[0].shape

    def matvec(self, v):
        return sum(Q.dot(v) for Q in self.matrices)

    def transpose(self):
        return Sum([Q.T for Q in self.matrices])

class VStack(EkteloMatrix):
    def __init__(self, matrices):
        # GDB: this needs to implement ndim as a member in order to support "*"
        # all matrices must have same number of columns
        self.matrices = matrices
        m = sum(Q.shape[0] for Q in matrices)
        n = matrices[0].shape[1]
        self.shape = (m,n)
    
    def matvec(self, v):
        return np.concatenate([Q.dot(v) for Q in self.matrices])

    def transpose(self):
        return HStack([Q.T for Q in self.matrices])

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
    
    def transpose(self):
        return VStack([Q.T for Q in self.matrices])

    def __abs__(self):
        return HStack([Q.__abs__() for Q in self.matrices])

def Kronecker(EkteloMatrix):
    def __init__(self, matrices):
        self.matrices = matrices
        self.shape = tuple(np.prod([Q.shape for Q in matrices]))
        self.dtype = matrices[0].dtype

    def matvec(self, v):
        size = self.shape[1]
        X = v
        for Q in self.matrices[::-1]:
            m, n = Q.shape
            X = Q @ X.reshape(size//n, n).T
            size = size * m // n
        return X.flatten()

    def transpose(self, v):
        return Kronecker([Q.T for Q in self.matrices]) 
   
    def gram(self):
        return Kronecker([Q.gram() for Q in matrices])
 
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
    pass
    #A = EkteloMatrix(np.eye(5))
    #print(np.eye(5).dtype)
