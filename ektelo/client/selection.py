# non-private query selection operators
from __future__ import division
from builtins import map
from builtins import zip
from builtins import range
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
import itertools
import math
from ektelo import util
from ektelo import support
from ektelo.operators import SelectionOperator
from functools import reduce

def flatten_measurements(m, dsize, sparse_flag = 1):
    ''' Given a list of coordinates (each row corresponds to one query)
        it returns a set of measurements in the desired format.
        sparse: flag that denotes if the returnred measurements are
                in an np.array format or scipy.sparse format
    '''
    all_measurements = []

    for measurement in m:
        M = np.zeros(dsize).astype(np.int8)
        flat_indices = measurement.flatten()
        M[flat_indices] = 1
        if sparse_flag == 1:
            M = sparse.csr_matrix(M)
        all_measurements.append(M)
    if sparse_flag == 1:
        s = sparse.vstack(all_measurements, format = 'csr')
    else:
        s = np.array(all_measurements)

    return s


def rect_to_quads(x):
    '''
    Given an np array it splits it correctly to 4 quads in the midpoints
    can handle arrays of arbitrary shape (1D as well)
    '''

    n_rows = x.shape[0]
    n_cols = x.shape[1]
    # If ncol is odd, do vert splits in balanced manner
    col_parity = 0
    if n_cols % 2:
        col_parity = 1
    col_midpoint = util.old_div(x.shape[1],2)
    row_midpoint = util.old_div(x.shape[0],2)

    if x.shape[0] == 1:
        # if x has only one row then do only vertical split
        x1, x2 = np.split(x, [col_midpoint], axis = 1)
        return [x1, x2]

    if x.shape[1] == 1:
        # if x has only one col then do only horizontal split
        x1, x2 = np.split(x, [row_midpoint], axis = 0)
        return [x1, x2]

    # o/w do both splits
    x_h1, x_h2 = np.split(x, [row_midpoint], axis = 0)
    x1, x2 = np.split(x_h1,  [col_midpoint], axis = 1)
    x3, x4 = np.split(x_h2,  [col_midpoint + col_parity], axis = 1)

    return [x1, x2, x3, x4]


def variance(N, b):
    '''Computes variance given domain of size N
    and branchng factor b.  Equation 3 from paper.'''
    h = math.ceil(math.log(N, b))

    return ( ((b - 1) * h**3) - (util.old_div((2 * (b+1) * h**2), 3)))


def buildHierarchical_ios(n, b):
    '''
    Does the same with buildHierarchical with the following differences:
        - Does not support different branching factors per level
        - Supports domain sizes that are not powers of the branching factor
    Note (ios): tested the equivalence between buildHierarchical_ios and buildHierarchical
    on domain sizes in [2, 4, 8, ..., 8192] for b = 2. Both functions produce the same set of queries.
    The new function is up to x4 faster.
    '''
    nodes       = {}
    root        = list(range(n))
    n_id        = 0
    nodes[n_id] = root
    pending     = [n_id]
    cur_id      = 0

    while len(pending) != 0:
        # Pop an ID from pending
        cur_id  = pending.pop()
        node    = nodes[cur_id]
        children = split_b_ways(node, b)
        for child in children:
            n_id         += 1
            nodes[n_id]  = child
            if len(child) > 1:
                pending.append(n_id)
    # Post process for the correct format
    tree = list(nodes.values())
    M = []
    for node in tree:
        m = np.zeros(n)
        m[node] = 1
        M.append(m)

    return np.array(M).astype(int)


def buildHierarchical_sparse(n, b):
    '''
    Builds a sparsely represented (csr_matrix) hierarchical matrix
    with n columns and a branching factor of b.  Works even when n
    is not a power of b
    '''
    if n == 1:
        return sparse.csr_matrix([1.0])
    if n <= b:
        a = np.ones(n)
        b = sparse.identity(n, format='csr')
        return sparse.vstack([a, b])

    # n = mb + r where r < b
    # n = (m+1) r + m (b-r)
    # we need r hierarchical matrices with (m+1) cols
    # and (b-r) hierarchical matrices with m cols
    m, r = divmod(n, b)
    hier0 = buildHierarchical_sparse(m, b) # hierarchical matrix with m cols
    if r > 0:
        hier1 = buildHierarchical_sparse(m+1, b) # hierarchical matrix with (m+1) cols

    # sparse.hstack doesn't work when matrices have 0 cols
    def hstack(left, hier, right):
        if left.shape[1] > 0 and right.shape[1] > 0:
            return sparse.hstack([left, hier, right])
        elif left.shape[1] > 0:
            return sparse.hstack([left, hier])
        else:
            return sparse.hstack([hier, right])

    res = [np.ones(n)]
    for i in range(r):
        rows = hier1.shape[0]
        start = (m+1)*i
        end = start + m+1
        left = sparse.csr_matrix((rows, start))
        right = sparse.csr_matrix((rows, n-end))
        res.append(hstack(left, hier1, right))
    for i in range(r, b):
        # (m+1) r + m (b-r) = (m+1) r + m (b-i) + m (i-r)
        rows = hier0.shape[0]
        start = (m+1)*r + m*(i-r)
        end = start + m
        left = sparse.csr_matrix((rows, start))
        right = sparse.csr_matrix((rows, n-end))
        res.append(hstack(left, hier0, right))

    return sparse.vstack(res, format='csr')


def find_best_branching(N):
    '''
    Technique from Qardaji et al. PVLDB 2013.
    Try all branchings from 2 to N and pick one
    with minimum variance.
    N in this context is domain size
    '''
    min_v = float('inf')
    min_b = None
    for b in range(2,N+1):
        v = variance(N, b)
        if v < min_v:
            min_v = v
            min_b = b

    return min_b


def Hb2D(n, m, b_h, b_v, sparse = 1):
    ''' Implementation of Hb for 2D histograms
            (n,m): the shape of x
            b_v, b_h = the vertical and horizontal branching factorr respectively
            sparse: flag that denotes if the returnred measurements are
                in an np.array format or scipy.sparse format
    '''
    dsize = n * m
    x = np.arange(dsize).reshape(n, m)

    # Avoid doing recursion, python is notoriously bad at it
    pending         = [x]
    measurement_coo = []
    while len(pending) != 0:
        cur_rect = pending.pop()
        # if it's a leaf then we don't want to split it anymore
        if cur_rect.shape[0] * cur_rect.shape[1] > 1:
            # split the current rectangle to rectangles according to the branching factors
            sub_rects = support.split_rectangle(cur_rect, b_v, b_h)
            for rect in sub_rects:
                measurement_coo.append(rect)
                pending.append(rect)

    # Flatten the measurements to correspond to the flattened x vector
    M = flatten_measurements(measurement_coo, dsize, sparse)

    return M


def quadtree(n, m, sparse = 1):
    ''' Quadtree function, accepts a shape (n, m)
        and returns a set of measurements in sparse format on the expanded x vector
        n and m can be arbitrary numbers
        sparse: flag that denotes if the returnred measurements are
                in an np.array format or scipy.sparse format
    '''
    dsize = n * m
    x = np.arange(dsize)
    x = x.reshape(n,m)

    # Avoid doing recursion, python is notoriously bad at it
    pending        = [x]
    measurement_coo = [x]
    while len(pending) != 0:
        cur_quad = pending.pop()

        # if it's a leaf then we don't want to split it anymore
        if cur_quad.shape[0] * cur_quad.shape[1] > 1:
            sub_quads = rect_to_quads(cur_quad)
            for quad in sub_quads:
                measurement_coo.append(quad)
                pending.append(quad)

    # Flatten the measurements to correspond to the flattened x vector
    M = flatten_measurements(measurement_coo, dsize, sparse)

    return M


def GenerateCells(n,m,num1,num2,grid):
    # this function used to generate all the cells in UGrid
    assert math.ceil(util.old_div(n,float(grid))) == num1 and math.ceil(util.old_div(m,float(grid))) == num2, "Unable to generate cells for Ugrid: check grid number and grid size"
    cells = []
    for i in range(num1):
        for j in range(num2):
            lb = [int(i*grid),int(j*grid)]
            rb = [int((i+1)*grid-1),int((j+1)*grid-1)]
            if rb[0] >= n:
                rb[0] = int(n-1)
            if rb[1] >= m:
                rb[1] = int(m-1)

            cells = cells + [[lb,rb]]

    return cells


def cells_to_query(cells,domain):
    '''
    helper function
    :param cells: UGrid cells represented as upper-left and lower-right coordinates(inclusive range)
    :param domain: tuple indicating the domain of queries.
    :return: workload represented as a sparse matrix, each row is a query with the flattened domain.
    '''
    query_number = len(cells)
    domain_size = np.prod(domain)
    matrix = sparse.lil_matrix((query_number, domain_size))

    query_no = 0
    for ul,lr in cells:
        up, left = ul; low, right = lr
        for row in range(up,low+1):
            begin, end = np.ravel_multi_index([[row,row],[left,right]],domain)
            matrix[query_no, begin:end+1] = [1]*(end-begin+1)
        query_no+=1

    # convert to csr format for fast arithmetic and matrix vector operations
    matrix = matrix.tocsr()

    return matrix


class Identity(SelectionOperator):

    def __init__(self, domain_shape):
        super(Identity, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, "Identity selection only workss for 1D"
        self.domain_shape = domain_shape

    def select(self):
        return sparse.identity(self.domain_shape[0])


class Total(SelectionOperator):

    def __init__(self, domain_shape):
        super(Total, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, "Total selection only works for 1D"
        self.domain_shape = domain_shape

    def select(self):
        return np.ones((1, self.domain_shape[0]), dtype=np.float)


class H2(SelectionOperator):
    """
    H2 select operator(1D): 
    Adds hierarchical queries with uniform branching factor.
    Works in an iterative top-down fashion splits each node V to create max{b, |V|} children and stops
    when a node has 1 element. This creates a balanced tree with maximum number of nodes at each level
    except (possibly) the last one.

    """

    def __init__(self, domain_shape, branching=2, matrix_form='sparse'):
        super(H2, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, 'Hierarchical selection only supports 1D and 2D domain shapes'
        assert branching > 1
        self.branching = branching
        self.matrix_form = matrix_form
        self.domain_shape = domain_shape
        self.cache = np.ones((1, 1))

    def select(self):
        if self.domain_shape[0] == self.cache.shape[1]:
            h_queries = self.cache
        elif self.matrix_form == 'dense':
            h_queries = buildHierarchical_ios(self.domain_shape[0], self.branching)
        else:
            h_queries = buildHierarchical_sparse(self.domain_shape[0], self.branching)

        return h_queries


class HB(SelectionOperator):
    '''
    HB select operator(1D and 2D)
    Add hierarchical queries with optimal branching factor, per Qardaji et al.
    '''

    def __init__(self, domain_shape, sparse_flag=1):
        super(HB, self).__init__()

        assert (isinstance(domain_shape, tuple) and len(domain_shape) == 1
                or len(domain_shape) == 2
                ), 'HB selection only supports 1D and 2D domain shapes'
        self.domain_shape = domain_shape
        self.sparse_flag = sparse_flag

    def select(self):
        if len(self.domain_shape) == 1:

            N = self.domain_shape[0]
            branching = find_best_branching(N)
            # remove root
            h_queries = buildHierarchical_sparse(N, branching).tocsr()[1:]

        elif len(self.domain_shape) == 2:
            N = self.domain_shape[0] * self.domain_shape[1]
            branching = find_best_branching(N)
            h_queries = Hb2D(self.domain_shape[0], 
                             self.domain_shape[1],
                             branching, 
                             branching, 
                             self.sparse_flag)
        return h_queries


class GreedyH(SelectionOperator):

    def __init__(self, domain_shape, W, branch=2, granu=100):
        super(GreedyH, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, 'greedyH selection only supports 1D  domain shapes'
        self.domain_shape = domain_shape
        self.W = W
        self._branch = branch
        self._granu = granu

    def select(self):
        if not isinstance(self.W, np.ndarray):
            W = self.W.toarray()
        else:
            W = self.W
        QtQ = np.dot(W.T, W)
        n = self.domain_shape[0]
        err, inv, weights, queries = self._GreedyHierByLv(
            QtQ, n, 0, withRoot=False)

        # form matrix from queries and weights
        row_list = []
        for q, w in zip(queries, weights):
            if w > 0:
                row = np.zeros(self.domain_shape[0])
                row[q[0]:q[1] + 1] = w
                row_list.append(row)
        mat = np.vstack(row_list)
        mat = sparse.csr_matrix(mat) if sparse.issparse(mat) is False else mat

        return mat

    def _GreedyHierByLv(self, fullQtQ, n, offset, depth=0, withRoot=False):
        """Compute the weight distribution of one node of the tree by minimzing
        error locally.

        fullQtQ - the same matrix as QtQ in the Run method
        n - the size of the submatrix that is corresponding
            to current node
        offset - the location of the submatrix in fullQtQ that
                 is corresponding to current node
        depth - the depth of current node in the tree
        withRoot - whether the accurate root count is given

        Returns: error, inv, weights, queries
        error - the variance of query on current node with epsilon=1
        inv - for the query strategy (the actual weighted queries to be asked)
              matrix A, inv is the inverse matrix of A^TA
        weights - the weights of queries to be asked
        queries - the list of queries to be asked (all with weight 1)
        """
        if n == 1:
            return np.linalg.norm(fullQtQ[:, offset], 2)**2, \
                np.array([[1.0]]), \
                np.array([1.0]), [[offset, offset]]

        QtQ = fullQtQ[:, offset:offset + n]
        if (np.min(QtQ, axis=1) == np.max(QtQ, axis=1)).all():
            mat = np.zeros([n, n])
            mat.fill(util.old_div(1.0, n**2))
            return np.linalg.norm(QtQ[:, 0], 2)**2, \
                mat, np.array([1.0]), [[offset, offset + n - 1]]

        if n <= self._branch:
            bound = list(zip(list(range(n)), list(range(1, n + 1))))
        else:
            rem = n % self._branch
            step = util.old_div((n - rem), self._branch)
            swi = (self._branch - rem) * step
            sep = list(range(0, swi, step)) + list(range(swi, n, step + 1)) + [n]
            bound = list(zip(sep[:-1], sep[1:]))

        serr, sinv, sdist, sq = list(zip(*[self._GreedyHierByLv
                                      (fullQtQ, c[1] - c[0], offset + c[0],
                                       depth=depth + 1) for c in bound]))
        invAuList = [c.sum(axis=0) for c in sinv]
        invAu = np.hstack(invAuList)
        k = invAu.sum()
        m1 = sum(map(lambda rng, v:
                     np.linalg.norm(
                         np.dot(QtQ[:, rng[0]:rng[1]], v), 2)**2,
                     bound, invAuList))
        m = np.linalg.norm(np.dot(QtQ, invAu), 2)**2
        sumerr = sum(serr)

        if withRoot:
            return sumerr, block_diag(*sinv), \
                np.hstack([[0], np.hstack(sdist)]), \
                [[offset, offset + n - 1]] + list(itertools.chain(*sq))

        decay = util.old_div(1.0, (self._branch**(util.old_div(depth, 2.0))))
        err1 = np.array(list(range(self._granu, 0, -1)))**2
        err2 = np.array(list(range(self._granu)))**2 * decay
        toterr = 1.0 / err1 * \
            (sumerr - ((m - m1) * decay + m1) * err2 / (err1 + err2 * k))

        err = toterr.min() * self._granu**2
        perc = 1 - util.old_div(np.argmin(toterr), float(self._granu))
        inv = (util.old_div(1.0, perc))**2 * (block_diag(*sinv)
                                 - (1 - perc)**2 / (perc**2 + k * (1 - perc)**2)
                                 * np.dot(invAu.reshape([n, 1]), invAu.reshape([1, n])))
        dist = np.hstack([[1 - perc], perc * np.hstack(sdist)])
        return err, inv, dist, \
            [[offset, offset + n - 1]] + list(itertools.chain(*sq))


class QuadTree(SelectionOperator):

    def __init__(self, domain_shape, sparse_flag=1):
        super(QuadTree, self).__init__()
        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 2, "QuadTree selection only workss for 2D"
        self.domain_shape = domain_shape
        self.sparse_flag = sparse_flag

    def select(self):
        strategy = quadtree(self.domain_shape[0], self.domain_shape[1], self.sparse_flag)
        return strategy


class UniformGrid(SelectionOperator):

    def __init__(self, domain_shape, data_sum, esp_par, ag_flag=False, c=10, gz=0):
        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 2, "UniformGrid selection only workss for 2D"
        super(UniformGrid, self).__init__()

        self.domain_shape = domain_shape
        self.data_sum = data_sum  # sum of x, assumed to be public
        # epsilon used as a paramter to calculate grid size, not consumed
        self.eps_par = esp_par
        self.c = c
        self.gz = gz
        # when ug is used as the first level of ag, calculation of grid size 
        # is slightly different. 
        self.ag_flag = ag_flag 

    def select(self):
        n, m = self.domain_shape
        N = self.data_sum
        eps = self.eps_par

        if self.ag_flag:
            m1 = int(math.sqrt((N*eps) / self.c) / 4 - 1) + 1
            if m1 < 10:
                m1 = 10
            M = m1**2
    
            grid = int(math.sqrt(n*m*1.0/M)-1)+1
            if grid <= 0:
                grid = 1

        else:
            M = util.old_div((N * eps), self.c)
            if self.gz == 0:
                grid = int(math.sqrt(n * m / M) - 1) + 1
            else:
                grid = int(self.gz)
            if grid < 1:
                grid = 1

        num1 = int(util.old_div((n - 1), grid) + 1)
        num2 = int(util.old_div((m - 1), grid) + 1)

        # TODO: potential optimization if grid ==1 identity workload
        cells = GenerateCells(n, m, num1, num2, grid)
        matrix = cells_to_query(cells, (n, m))

        return matrix


class AdaptiveGrid(SelectionOperator):

    def __init__(self, domain_shape, x_hat, eps_par, c2=5):
        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 2, "AdptiveGrid selection only works for 2D"
        super(AdaptiveGrid, self).__init__()

        self.domain_shape = domain_shape
        self.x_hat = x_hat
        # epsilon used as a paramter to calculate grid size, not consumed
        self.eps_par = eps_par
        self.c2 = c2

    def select(self):

        shape = self.domain_shape

        if shape == (1, 1):
            # skip the calucation of newgrids if shape is of size 1
            matrix = sparse.csr_matrix(([1], ([0], [0])), shape=(1, 1))
            newgrid = 1

        else:
            eps = self.eps_par
            cur_noisy_x = self.x_hat
            noisycnt = cur_noisy_x.sum()
            # compute second level grid size
            if noisycnt <= 0:
                m2 = 1
            else:
                m2 = int(math.sqrt(noisycnt * eps / self.c2) - 1) + 1
            M2 = m2**2
            nn, mm = shape
            newgrid = int(math.sqrt(nn * mm * 1.0 / M2) - 1) + 1
            if newgrid <= 0:
                newgrid = 1
            num1 = int(util.old_div((nn - 1), newgrid) + 1)
            num2 = int(util.old_div((mm - 1), newgrid) + 1)
            # generate cell and pending queries base on new celss
            cells = GenerateCells(nn, mm, num1, num2, newgrid)
            matrix = cells_to_query(cells, (nn, mm))

        return matrix


class Wavelet(SelectionOperator):
    '''
    Adds wavelet matrix as measurements
    '''
    #TODO: handle 2D

    def __init__(self, domain_shape):
        super(Wavelet, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, 'Wavelet selection only supports 1D  domain shapes'

        self.domain_shape = domain_shape


    @staticmethod
    def wavelet_sparse(n):
        '''
        Returns a sparse (csr_matrix) wavelet matrix of size n = 2^k
        '''
        if n == 1:
            return sparse.identity(1, format='csr')
        m, r = divmod(n, 2)
        assert r == 0, 'n must be power of 2'
        H2 = Wavelet.wavelet_sparse(m)
        I2 = sparse.identity(m, format='csr')
        A = sparse.kron(H2, [1,1])
        B = sparse.kron(I2, [1,-1])
        return sparse.vstack([A,B])

    @staticmethod
    def remove_duplicates(a):
        ''' 
        Removes duplicate rows from a 2d numpy array
        '''
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))[::-1]

    @staticmethod
    def power(n, b):
        '''
        Helper function for domain sizes and branching
        Returns exponent exp such that n == b ** exp when n is a power of b
        Otherwise returns closest exp and remainder, so that n == b ** exp + rem
        '''
        exp = int(math.log(n, b))
        rem = n - (b ** exp)
        return exp, rem
    
    @staticmethod
    def wavelets(n):
        """
        can deal with domain sizes that are not powers of 2
        """
        height, rem = Wavelet.power(n, 2)
        diff = 0
        if rem != 0:
            height += 1
            diff    = 2 ** (height) - n
            n       = 2 ** (height)
        M = [np.ones(n)]
        step = 2*n
        for h in range(height):
            n_nodes = 2 ** h
            step //= 2
            for node in range(n_nodes):
                x     = np.zeros(n)
                start = node * step
                mid   = int ((node + 0.5) * step)
                end   = (node + 1)   * step

                x[start:mid] =  1
                x[mid:end]   = -1
                M.append(x)
        M = np.delete(M, np.arange(diff) + n - diff ,  axis = 1)
        if rem!= 0:
            M = Wavelet.remove_duplicates(M)
        return M


    def select(self):
        n = self.domain_shape[0]

        if (n != 0 and ((n) & (n-1)) ==0):
            # if power of 2
            wavelet_query = Wavelet.wavelet_sparse(n)
        else:
            wavelet_query = Wavelet.wavelets(n)

        return wavelet_query


class AddEquiWidthIntervals(SelectionOperator):
    """
    Given a selected query, the first row in W, it complements select query with a set of disjoint interval queries, of
    a specified width.
    """
    def __init__(self, W, log_width):
        super(AddEquiWidthIntervals, self).__init__()
        self.M_hat = support.extract_M(W)   # expects W to contain a single measurement
        self.grid_size = min(2 ** log_width, self.M_hat.shape[1])

    def select(self):
        return sparse.vstack((self.M_hat, support.complement(self.M_hat, self.grid_size)))

class HDMarginal(SelectionOperator):

    def __init__(self, domain_shape):
        super(HDMarginal, self).__init__()

        self.domain_shape = domain_shape

    def select(self):
        domain_shape = self.domain_shape
        marginals = []
        for ind,shape in enumerate(domain_shape):
            queries = [np.ones(n) for n in domain_shape[:ind]] + [sparse.identity(shape)] + [np.ones(n) for n in domain_shape[ind+1:]]
            queries = reduce(sparse.kron, queries)
            marginals.append(queries)
        strategy = sparse.vstack(marginals)

        return strategy

