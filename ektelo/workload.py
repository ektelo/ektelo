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

    def get_matrix(self, matrix_format = 'sparse'):
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

class Identity(Workload, CartesianInstantiable):
    """ Identity workload for in k-dimensional domain """

    def __init__(self, domain_shape, weight=1.0, pretty_name='identity'):
        self.init_params = util.init_params_from_locals(locals())
        self.weight = weight
        self.pretty_name = pretty_name
        if type(domain_shape) is int:
            domain_shape = (domain_shape,)
        indexes = itertools.product(*[list(range(i)) for i in domain_shape])   # generate index tuples
        queries = [ndRangeUnion().addRange(i,i,weight) for i in indexes]
        super(self.__class__,self).__init__(queries, domain_shape)

    def compute_matrix_sparse(self):
        return sparse.identity(numpy.prod(self.domain_shape))

    @classmethod
    def oneD(cls, domain_shape_int, weight=1.0):
        return cls((domain_shape_int,), weight)

    @staticmethod
    def instantiate(params):
        return Identity(params['domain'])

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(str(self.weight)))
        m.update(util.prepare_for_hash(str(util.standardize(self.domain_shape))))
        return m.hexdigest()

class Total(Workload, CartesianInstantiable):
    def __init__(self, domain_shape, pretty_name = 'total'):
        self.pretty_name = pretty_name
        if type(domain_shape) is int:
            domain_shape = (domain_shape,)
        lb = tuple(0 for _ in domain_shape)
        ub = tuple(x-1 for x in domain_shape)
        q = ndRangeUnion().addRange(lb, ub, 1.0)
        super(self.__class__, self).__init__([q], domain_shape)

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(str(util.standardize(self.domain_shape))))
        return m.hexdigest()
        
class Concatenate(Workload):
    """
    A class for constructing workloads of the same shape by concatenating their queries.
    """
    def __init__(self, workloads, pretty_name = 'concat'):
        """
        :param workloads: a list of workoads defined on the same domain
        :param pretty_name: name of this workload (default = 'concat')
        :returns: A new workload with all the queries from the given workloads
        
        Note: this class provides an efficient implementation of W.get_matrix
        for matrix_format='sparse', 'dense', and 'linop' assuming there is an
        efficient implementation in each of the given workloads.
        """
        self.pretty_name = pretty_name
        self.workloads = workloads
        assert len(workloads) >= 1, 'must have at least 1 workload'
        domain_shape = workloads[0].domain_shape
        assert all(w.domain_shape == domain_shape for w in workloads), 'shape mismatch'
        query_list = []
        for w in workloads:
            query_list.extend(w.query_list)
        super(Concatenate, self).__init__(query_list, domain_shape)

    # TODO(ryan): if the purpose of overriding is for efficiency
    # should we even worry about dense matrices?
    def compute_matrix_dense(self):
        matrices = [w.compute_matrix_dense() for w in self.workloads]
        return numpy.vstack(matrices)

    # overridden for efficiency
    def compute_matrix_sparse(self):
        matrices = [w.compute_matrix_sparse() for w in self.workloads]
        return sparse.vstack(matrices).tocsr() # TODO(ryan): should we convert to csr?

    def compute_matrix_linop(self):
        linops = [w.compute_matrix_linop() for w in self.workloads]
        sizes = [L.shape[0] for L in linops]
        indices = numpy.cumsum(sizes)
        n = numpy.prod(self.domain_shape)
        m = sum(sizes)
        def matvec(x):
            return numpy.concatenate([L.matvec(x) for L in linops])
        def rmatvec(x):
            return sum(L.rmatvec(v) for L,v in zip(linops, numpy.split(x, indices)))
        return linalg.LinearOperator((m,n), matvec, rmatvec, dtype=numpy.float64)

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        for w in self.workloads:
            m.update(util.prepare_for_hash(w.hash))
        return m.hexdigest()

class Kronecker(Workload):
    """
    A class for constructing high dimensional workloads from low dimensional building blocks.

    Roughly speaking, this class constructs a new workload by taking the cartesian product of 
    queries from the input workloads.
    """
    def __init__(self, workloads, pretty_name = 'kronecker'):
        """
        :param workloads: a list of workoads defined on any combination of domains
        :param pretty_name: name of this workload (default = 'kronecker')
        :returns: A new high dimensional workload constructed by taking the
            cartesian product of queries from the given workloads
        
        Note: this class provides an efficient implementation of W.get_matrix
        for matrix_format='sparse', 'dense', and 'linop' assuming there is an
        efficient implementation in each of the given workloads.  Thus it is
        preferable to use this over the (lowercase) function kronecker.
        """
        self.pretty_name = pretty_name
        self.workloads = workloads
        this = kronecker(workloads)
        super(Kronecker, self).__init__(this.query_list, this.domain_shape)

    def compute_matrix_dense(self):
        matrices = [w.compute_matrix_dense() for w in self.workloads]
        return reduce(numpy.kron, matrices)

    # overridden for efficiency
    def compute_matrix_sparse(self):
        matrices = [w.compute_matrix_sparse() for w in self.workloads]
        return reduce(sparse.kron, matrices)
        
    # overridden for efficiency
    # note that the linear operator will always be space efficient
    # it will be most time efficient if the true LinearOperators are 
    # on the last dimension (sparse matrices come first)
    # if workloads = [S,S,S,L,S,S,S,L] --> [S,S,S,L,L,L,L,L] (bad)
    # if workloads = [S,S,S,S,S,S,L,L] --> [S,S,S,S,S,S,L,L] (good)
    # since reduce is left associative, the first linear operator it sees 
    # will cause all subsequent objects to be linear operators too
    # kron for linear operators is lazy, so there is some overhead to that
    def compute_matrix_linop(self):
        linops = [w.compute_matrix_linop() for w in self.workloads]
        return reduce(linop_kron, linops) 

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        for w in self.workloads:
            m.update(util.prepare_for_hash(w.hash))
        return m.hexdigest()

class KWayMarginals(Concatenate, CartesianInstantiable):
    """ Workload consisting of all K-way marginals where 0 <= k <= number of dimensions"""

    def __init__(self, domain_shape, k, pretty_name = 'k-way marginals'):
        self.init_params = util.init_params_from_locals(locals())
        self.pretty_name = pretty_name
        if type(domain_shape) is int:
            domain_shape = (domain_shape,)
        assert 0 <= k <= len(domain_shape), 'invalid k'
        self.k = k
        D = len(domain_shape)

        idents = [Identity.oneD(n) for n in domain_shape]
        totals = [Total(n) for n in domain_shape]

        workloads = []
        for c in itertools.combinations(list(range(D)), k):
            oned = [idents[i] if i in c else totals[i] for i in range(D)]
            workloads.append(Kronecker(oned))
        super(self.__class__, self).__init__(workloads)
    
    @staticmethod
    def instantiate(params):
        return KWayMarginals(params['domain'],params['k'])

    @property 
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(str(self.k)))
        m.update(util.prepare_for_hash(str(util.standardize(self.domain_shape))))
        return m.hexdigest()

class PrefixMarginals(Kronecker, CartesianInstantiable):
    """ Workload consisting of Prefix queries on dimensions identified by prefix_axes
        and marginals on all other axes
    """

    def __init__(self, domain_shape, prefix_axes=(0,), pretty_name='prefix marginals'):
        self.init_params = util.init_params_from_locals(locals())
        self.pretty_name = pretty_name
        self.prefix_axes = prefix_axes
        if type(domain_shape) is int:
            domain_shape = (domain_shape,)
        
        if len(domain_shape) - len(prefix_axes) > min(prefix_axes):
            import warnings
            warnings.warn('It is not recommended to construct PrefixMarginals this way.  Transpose the data so that the prefix workloads are on the last dimensions')
            

        oned = []
        for i, n in enumerate(domain_shape):
            if i in self.prefix_axes:
                oned.append( Prefix1D(n) )
            else:
                oned.append(Identity.oneD(n) + Total(n))
        super(self.__class__, self).__init__(oned)

    @staticmethod
    def instantiate(params):
        return PrefixMarginals(params['domain'], params['prefix_axes'])

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(str(util.standardize(self.domain_shape))))
        m.update(util.prepare_for_hash(str(util.standardize(self.prefix_axes))))
        return m.hexdigest()

def merge(q1, q2):
    ''' merge an n dimensional query and an m dimensional query into an (n+m) dimensional query '''
    ans = ndRangeUnion()
    for (lb1, ub1, wgt1) in q1.ranges:
        for (lb2, ub2, wgt2) in q2.ranges:
            ans.addRange(lb1 + lb2, ub1 + ub2, wgt1 * wgt2)
    return ans

def kron(W1, W2):
    ''' merge an n dimensional workload and an m dimensional workload into an (n+m) dimensional workl
oad '''
    dom1, dom2 = W1.domain_shape, W2.domain_shape
    queries = [merge(q1, q2) for q1 in W1.query_list for q2 in W2.query_list]
    return Workload(queries, dom1 + dom2)

# (Workload, kron) is a monoid with identity element zero
def kronecker(workloads):
    ''' merge a list of low dimensional workloads into a single high dimensional workload '''
    zq = ndRangeUnion().addRange(tuple(), tuple(), 1.0)
    zero = Workload([zq], tuple())
    return reduce(kron, workloads, zero)

def linop_kron(A, B):
    """
    Compute the kron product between two scipy.sparse.LinearOperators

    :param A: A m1 x n1 linear operator
    :param B: A m2 x n2 linear operator
    :returns: A new linear operator of size m1*m2 x n1*n2 representing the matrix kron(A,B)

    """
    # if A and B are both sparse matrices, we are better off explicitly computing kron product
    if isinstance(A, linalg.interface.MatrixLinearOperator) and \
        isinstance(B, linalg.interface.MatrixLinearOperator) and \
        sparse.isspmatrix(A.A) and sparse.issparse(B.A):
        return linalg.aslinearoperator(sparse.kron(A.A,B.A))

    am, an = A.shape
    bm, bn = B.shape
    shape = (am*bm, an*bn)
    # These expressions are derived from the matrix equations
    # vec(A X B) = kron(A, B^T) vec(X) where vec(Z) is Z.flatten()
    def matvec(x):
        X = x.reshape(an, bn, order='C')
        return B.matmat(A.matmat(X).T).T.flatten(order='C')
    def rmatvec(x):
        X = x.reshape(am, bm, order='C')
        return B.H.matmat(A.H.matmat(X).T).T.flatten(order='C')
    return linalg.LinearOperator(shape, matvec, rmatvec, dtype=numpy.float64)


    