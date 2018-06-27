import inspect

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
    # DelegateMatrix(s) * DelegateMatrix(s.T)
    #
    # Doesn't work:
    # m * m

    def __init__(self, mat):
        self._mat = mat
        self.local_members = [member[0] for member in inspect.getmembers(DelegateMatrix)] 
        self.delegate_members = [member[0] for member in inspect.getmembers(self._mat)]

    def __dir__(self):
        return sorted(set(self.local_members).union(set(self.delegate_members)))

    def __add__(self, other):
        print('adding', self, other)
        return self._mat + other._mat
    
    def __mul__(self, other):
        print('multiplying', self, other)
        return self._mat * other._mat
 
    def __abs__(self):
        print('absing', self)
        return self._mat.__abs__()
 
    def __getattr__(self, name):
        if name not in self.local_members:
            print('delegating', self, name)
            return getattr(self._mat, name)
            

