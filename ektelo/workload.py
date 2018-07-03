from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
import copy
import hashlib
import itertools
import numpy
from ektelo.mixins import Marshallable
from ektelo.mixins import CartesianInstantiable
from ektelo.query_nd_union import ndRangeUnion
from ektelo import util
from scipy import sparse
from scipy.sparse import linalg
from functools import reduce

"""
These classes define workloads
"""

class Workload(Marshallable):

    def __init__(self, query_list, domain_shape):
        """ Basic constructor takes list of ndQuery instances and a domain shape
        """
        self.domain_shape = domain_shape
        self.query_list = query_list
        self._matrix = {}

    def compile(self):
        return self

    def get_matrix(self, matrix_format = 'delegate_matrix'):
        """
        Produces the matrix represenation of this workload on the flatten domain.
        Note that W.evaluate(X) = W.get_matrix().dot(X.flatten()) where X is a ndarray 
        with shape = W.domain_shape.

        :param matrix_format: 'sparse' or 'dense' or 'linop'
            'sparse' --> for a scipy sparse matrix (e.g., csr, bsr)
            'dense' --> for a 2d numpy array
            'linop' --> for a scipy sparse LinearOperator 
        :returns: the matrix represenation of this workload

        Notes for Practical Usage:
        * All three return types support dot product and transpose dot product Wx and W^T x
        * It's best to never materialize the matrix unless it is necessary to do so
        * matrix_format='linop' should be used if the workload cannot be efficiently represented as a sparse matrix (e.g., prefix workload)
        * matrix_format='sparse' should be used if the workload is sparse (e.g., Identity, AllMarginals) and you need matrix operations that are not supported by 'linop' such as indexing
        * matrix_format='dense' should not be used unless the domain is small (<= 8192) 
        """
        if matrix_format in self._matrix:
            pass
        elif matrix_format == 'dense':
            self._matrix['dense'] = self.compute_matrix_dense()
        elif matrix_format == 'sparse':
            self._matrix['sparse'] = self.compute_matrix_sparse()
        elif matrix_format == 'linop':
            self._matrix['linop'] = self.compute_matrix_linop()
        elif matrix_format == 'delegate_matrix':
            from ektelo.math import DelegateMatrix
            self._matrix['delegate_matrix'] = DelegateMatrix(self.compute_matrix_sparse())

        return self._matrix[matrix_format]

    matrix = property(get_matrix)

    @property
    def size(self):
        return len(self.query_list)

    def __add__(self, other):
        """
        Add two workloads defined on the same domain by concatenating their queries

        :param other: the other workload to add
        :returns: a new workload with queries from this workload and other workload

        Notes: 
        * It is preferable to use the Concatenate class instead of this operation
        * Information about how to efficiently compute the workload matrix is lost
            when using this operation.
        * It's okay to use this operation if you will never need the workload matrix, 
            the domain size is small, or you prefer elegance to performance
        """

        assert self.domain_shape == other.domain_shape, 'domain shapes must match'
        return Workload(self.query_list + other.query_list, self.domain_shape)

    def compute_matrix_sparse(self):
        rows = len(self.query_list)
        cols = numpy.prod(self.domain_shape)
        flatidx = numpy.arange(cols).reshape(self.domain_shape)
        matrix = sparse.lil_matrix((rows, cols))
        for q, query in enumerate(self.query_list):
            for lb, ub, wgt in query.ranges:
                s = [slice(l, u+1) for l, u in zip(lb, ub)]
                idx = flatidx[s].flatten()
                matrix[q, idx] += numpy.repeat(wgt, idx.size)
        return matrix.tocsr()

    def compute_matrix_dense(self):
        rows = [r.asArray(self.domain_shape).flatten() for r in self.query_list]
        n = rows[0].size
        m = len(rows)
        matrix = numpy.empty(shape=(m,n))
        for (i,row) in enumerate(rows):
            matrix[i,:] = row
        return matrix

    # this is a default implemenation
    # subclasses can optionally make it more efficient
    def compute_matrix_linop(self):
        return linalg.aslinearoperator(self.matrix)

    def sensitivity(self):
        # copied from utilities.py
        ''' Compute sensitivity of a collection of ndRangeUnion queries '''
        maxShape = tuple( [max(l) for l in zip(*[q.impliedShape for q in self.query_list])] )
        array = numpy.zeros(maxShape)
        for q in self.query_list:
            array += q.asArray(maxShape)
        return numpy.max(array)

    def sensitivity_from_matrix(self):
        """Return the L1 sensitivity of input matrix A: maximum L1 norm of the columns."""
        return numpy.abs(self.matrix).sum(axis=0).max()

    def evaluate_matrix(self, x):
        return self.matrix.dot(x.ravel())

    def evaluate(self,x):
        '''evaluating the workload without materializing it
        :param x: data to be evaluated, expressed in flattened form
        :return : a list of queries answers'''
        assert numpy.prod(self.domain_shape) == x.size
        x= numpy.array(x, copy=True).reshape(self.domain_shape)
        return numpy.array([q.eval(x) for q in self.query_list])

    @property
    def key(self):
        """ Using leading 8 characters of hash as key for now """
        return self.hash[:8]

    def asDict(self):
        d = util.class_to_dict(self, ignore_list=['matrix', 'query_list', '_matrix'])
        return d

    def analysis_payload(self):
        return util.class_to_dict(self, ignore_list=['matrix', 'query_list', '_matrix'])

class Prefix1D(Workload, CartesianInstantiable):
    """ Workload of all 1D range queries with left bound equal to 0
        (Prefix is not well-defined in higher dimensions)
    """

    def __init__(self, domain_shape_int, pretty_name='prefix 1D'):
        self.init_params = util.init_params_from_locals(locals())

        self.pretty_name = pretty_name

        queries = [ndRangeUnion().add1DRange(0, c, 1.0) for c in range(domain_shape_int)]
        super(self.__class__,self).__init__(queries, (domain_shape_int,))

    def __repr__(self):
        r = self.__class__.__name__ + '('
        r += 'domain_shape_int=' + str(self.domain_shape[0]) + ')'
        return r

    def compute_matrix_linop(self):
        n = numpy.prod(self.domain_shape)
        def matvec(x):
            return x.cumsum()
        def rmatvec(x):
            return x[::-1].cumsum()[::-1].astype(numpy.float64)
        return linalg.LinearOperator((n,n), matvec, rmatvec, dtype=numpy.float64)

    @staticmethod
    def instantiate(params):
        return Prefix1D(params['domain'])

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(str(util.standardize(self.domain_shape))))
        return m.hexdigest()

class RandomRange(Workload, CartesianInstantiable):
    ''' Generate m random n-dim queries, selected uniformly from list of shapes and placed randomly in n-dim domain
        shape_list: list of shape tuples
        domain_shape: a shape tuple describing domain
        m: number of queries in result
        Note: for 1-dim, shapes must be unary tuples, e.g. (2,) (or see convenience method below)
    '''

    def __init__(self, shape_list, domain_shape, size, seed=9001, pretty_name='random range'):
        self.init_params = util.init_params_from_locals(locals())

        self.shape_list = copy.deepcopy(shape_list)
        self.seed = seed
        self.pretty_name = pretty_name
        self._size = size

        prng = numpy.random.RandomState(seed)
        if shape_list == None:
            shapes = randomQueryShapes(domain_shape, prng)
        else:
            prng.shuffle(self.shape_list)
            shapes = itertools.cycle(self.shape_list) # infinite iterable over shapes in shape_list
        queries = []
        for i in range(size):
            lb, ub = placeRandomly(next(shapes), domain_shape, prng)       # seed must be None or repeats
            queries.append( ndRangeUnion().addRange(lb,ub,1.0) )
        super(RandomRange,self).__init__(queries, domain_shape)

    @staticmethod
    def instantiate(params):
        domain = params['domain']

        try:
            int(params['domain'])
            domain = [domain]
        except:
            pass

        shape_list = params['shape_list'] if 'shape_list' in params else None

        return RandomRange(shape_list, domain, params['query_size'], params['work_seed'])

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(str(self._size)))
        m.update(util.prepare_for_hash(str(util.standardize(self.shape_list))))
        m.update(util.prepare_for_hash(str(util.standardize(self.domain_shape))))
        return m.hexdigest()

    @classmethod
    def oneD(cls, shape_list, domain_shape_int, size, seed=9001):
        ''' Convenience method allowing ints to be submitted in 1D case '''
        if shape_list == None:
            return cls(None,(domain_shape_int,), size, seed)
        return cls([(i,) for i in shape_list], (domain_shape_int,), size, seed)


def randomQueryShapes(domain_shape, prng):
    ''' Generator that produces a list of range shapes; can be passed as iterator
        domain_shape: is the shape tuple of the domain
        prng: is numpy RandomState object
    '''
    while True:
        shape = [prng.randint(1, dim+1, None) for dim in domain_shape]
        yield tuple(shape)


def placeRandomly(query_shape, domain_shape, prng=None):
    ''' Place a n-dim query randomly in n-dim domain
        Return lb tuple and ub tuple which can be used to construct a range query object
    '''
    if not prng:
        prng = numpy.random.RandomState()

    lb, ub = [], []
    for i, val in enumerate(query_shape):
        lower = prng.randint(0, domain_shape[i] - query_shape[i] + 1, None)
        lb.append(lower)
        ub.append(lower + query_shape[i] - 1)
    return tuple(lb), tuple(ub)
