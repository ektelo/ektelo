import inspect
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.linalg.interface
from scipy.sparse import spmatrix


def vstack(dmatrices, format=None, dtype=None):
    types = {type(mat) for mat in dmatrices}

    if len(types) == 1 and types.copy().pop() == DelegateMatrix:
        blocks = [dmat.tocsr() for dmat in dmatrices]

        return DelegateMatrix(scipy.sparse.vstack(blocks, format, dtype))
    elif DelegateMatrix not in types:
        return scipy.sparse.vstack(dmatrices)
    else:
        raise TypeError('cannot vstack DelegateMatrix with other type')


def diag_like(mat, data, diags, m, n, format=None):
	diag = scipy.sparse.spdiags(data, diags, m, n, format)

	if type(mat) == DelegateMatrix:
		return DelegateMatrix(diag)
	else:
		return diag


def sparse_like(mat):
	if type(mat) == DelegateMatrix:
		return DelegateMatrix(scipy.sparse.csr_matrix(mat._mat.shape))
	else:
		return scipy.sparse.csr_matrix(mat.shape)


class DelegateMatrix(scipy.sparse.linalg.LinearOperator):
    # import scipy.sparse.linalg; import scipy.sparse.linalg.interface; import numpy as np; from scipy import sparse; from ektelo.math import DelegateMatrix; s = sparse.csr_matrix((3, 3), dtype=np.int8); s[0,0] = 3; s[1,1] = 1; s[2,2] = 2; b = np.array([1,1,1]); m = DelegateMatrix(s)
    #
    # This approach is very flexible. Not only do you automatically inherit many methods on the delegate object
    # but numpy is also capable of handling duck-typed objects, therefore many builtin methods also work.
    # The principal drawback is that the return type of every call is that of the delegate object. So we
    # cannot easily do things like m * m.T because the first object with be a DelegateMatrix and the second 
    # will be a scipy sparse matrix.
    #
    # Works:
    # m.shape
    # m + m
    # np.abs(m)
    # np.sum(m)
    # m.T
    # m * m.T
    # m * s.T
    # scipy.sparse.linalg.lsmr(m, b)
    #
    # Doesn't work:
    # s * m.T

    def _matvec(self, b):
        raise NotImplementedError( "_matvec" )

    def __init__(self, mat):
        self._mat = mat

    def _wrap(self, result):
        if np.isscalar(result):
            return result
        else:
            return DelegateMatrix(result)

    def __abs__(self):
        return DelegateMatrix(self._mat.__abs__())

    def __add__(self, other):
        if type(other) == DelegateMatrix:
            other = other._mat
        else:
            raise TypeError('incompatible type %s for multiplication with DelegateMatrix' % type(other))

        return DelegateMatrix(self._mat + other)

    def __mul__(self, other):
        if type(other) == DelegateMatrix:
            other = other._mat

        if scipy.sparse.compressed._cs_matrix in type(other).mro():
            return DelegateMatrix(self._mat * other)
        elif type(other) == np.ndarray:
            return self._mat * other
        else:
            raise TypeError('incompatible type %s for multiplication with DelegateMatrix' % type(other))

    def dot(self, other):
        if type(other) == DelegateMatrix:
            other = other._mat

        return self._mat.dot(other)

    def matmat( self, x ):
        return self._mat * x

    def matvec( self, x ):
        return self._mat * x

    def max(self, axis=None, out=None):
        return self._wrap(self._mat.max(axis, out))

    def sum(self, axis=None, dtype=None, out=None):
        return self._wrap(self._mat.sum(axis, dtype, out))

    def rmatvec( self, x ):
        return self._mat.T * x

    def tocsr(self):
        return self._mat.tocsr()
 
    def toarray(self):
        return self._mat.toarray()
 
    def transpose(self, axes=None):
        return DelegateMatrix(self._mat.transpose(axes))

    @property
    def dtype(self):
        return self._mat.dtype

    @property
    def ndim(self):
        return self._mat.ndim
    
    @property
    def shape(self):
        return self._mat.shape

    @property
    def T(self):
        return self.transpose() 
