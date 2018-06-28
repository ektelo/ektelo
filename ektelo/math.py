import inspect
import numpy as np

class DelegateMatrix:
    # import numpy as np; from scipy import sparse; from ektelo.math import DelegateMatrix; s = sparse.csr_matrix((3, 4), dtype=np.int8); m = DelegateMatrix(s)
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
    #
    # Doesn't work:
    # s * m.T

    def __init__(self, mat):
        self._mat = mat

    def __abs__(self):
        print('absing', self)
        return self._mat.__abs__()

    def __add__(self, other):
        print('adding', self, other)
        if type(other) == DelegateMatrix:
            other = other._mat
        return DelegateMatrix(self._mat + other)

    def __mul__(self, other):
        print('multiplying', self, other)
        if type(other) == DelegateMatrix:
            other = other._mat
        return DelegateMatrix(self._mat * other)
 
    @property
    def dtype(self):
        return self._mat.dtype

    @property
    def ndim(self):
        return self._mat.ndim
    
    @property
    def T(self):
        return DelegateMatrix(self._mat.T)
 
    @property
    def shape(self):
        return self._mat.shape
