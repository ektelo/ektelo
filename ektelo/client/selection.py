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
import ektelo
from ektelo import util
from ektelo import support
from ektelo.operators import SelectionOperator
from ektelo import matrix, workload
from functools import reduce


def variance(N, b):
    '''Computes variance given domain of size N
    and branchng factor b.  Equation 3 from paper.'''
    h = math.ceil(math.log(N, b))

    return ( ((b - 1) * h**3) - (util.old_div((2 * (b+1) * h**2), 3)))


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


def split_rectangle(rect_range, b_h, b_v):
    """
    Split rectangular domain into grids according to branching factors
    Input and output ranges represented as corner coordinates (inclusive)
    For use with Hb2D, Ugird and Agrid
    """
    ul, lr = rect_range
    upper, left =  ul
    lower, right = lr   
    n_rows = lower - upper + 1
    n_cols = right - left + 1  

    h_split = b_h
    v_split = b_v

    if b_h > n_rows:
        h_split = n_rows
        boarder_h = [ (i, i) for i in range(h_split)]

    elif n_rows % b_h != 0:
        new_hsize = np.divide(float(n_rows), b_h)
        h_split = [np.ceil(new_hsize * (i + 1)).astype(int) for i in range(b_h - 1)]
        temp = [i -1 for i in h_split]
        boarder_h = list(zip(([0] + h_split), (temp + [n_rows-1])) )

    else:
        cell_size_h = util.old_div(n_rows, h_split)
        boarder_h = [(i * cell_size_h, (i+1) * cell_size_h - 1) for i in range(h_split)]


    if b_v > n_cols:
        v_split = n_cols
        boarder_v = [(i, i) for i in range(v_split)]
    elif n_cols % b_v != 0:
        new_vsize = np.divide(float(n_cols), b_v)
        v_split = [np.ceil(new_vsize * (i + 1)).astype(int) for i in range(b_v - 1)]
        temp = [i -1 for i in v_split]
        boarder_v = list(zip(([0] + v_split), (temp + [n_cols-1])) )

    else:
        cell_size_v = util.old_div(n_cols, v_split)
        boarder_v = [(i * cell_size_v, (i+1) * cell_size_v - 1) for i in range(v_split)]


    lower_list, upper_list = [], []
    for (hl,hh), (vl, vh) in itertools.product(boarder_h, boarder_v):
        lower_list.append((upper + hl, left + vl))
        upper_list.append((upper + hh, left + vh))

    return lower_list, upper_list

def same_shape(rect_l, rect_u):
    '''
    Check the shape of all candidate rects,
    return the indices in groups with the same shape
    '''
    dic = dict()
    shape = np.array(rect_u) - np.array(rect_l)
    for i in range(len(shape)):
        s = tuple(shape[i])
        if s in dic:
            dic[s].append(i)
        else:
            dic[s] = [i]
    return list(dic.values())

def quick_product_sum(*arrays):
    '''
    Quick calculation of cross products of a list of arrays.
    Return the sum of each result
    '''
    la = len(arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype='int')
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    M = arr.reshape(la, -1)
    return M.sum(axis = 0)

def expand_offsets(cur_rect_l, cur_rect_u, offsets_h, offsets_v):
    '''
    Expand offsets at different level to generate the final offsets for 
    all candidate by computing the sum of each tuple in the cross product 
    of offset arrays.
    e.g two level offsets [[0, 1, 0], [2, 4, 2]] will be expanded to 
    [2 4 2 3 5 3 2 4 2]
    '''
    # Remove starting empty offset
    offsets_h = offsets_h[1:] if offsets_h[0] == [] else offsets_h
    offsets_v = offsets_v[1:] if offsets_v[0] == [] else offsets_v

    if len(offsets_h) == 0 and len(offsets_v) == 0:
        return [], []
    horizontal_offsets = quick_product_sum(*offsets_h)
    vertical_offsets   = quick_product_sum(*offsets_v)
    return np.vstack((horizontal_offsets + cur_rect_l[0], cur_rect_l[1] + vertical_offsets)).T, \
           np.vstack((cur_rect_u[0] + horizontal_offsets, cur_rect_u[1] + vertical_offsets)).T


def Hb2D(n, m, b_h, b_v):
    ''' Implementation of Hb for 2D histograms
            (n,m): the shape of x
            b_v, b_h = the vertical and horizontal branching factorr respectively
    '''
    # Avoid doing recursion, python is notoriously bad at it
    pending_l, pending_u = [ (0, 0) ], [(n - 1, m - 1)]
    # Avoid calculating partition for quads of the same shape
    # by storing a base quad and a list of offset in each level of the hierarchy 
    pending_offsets_h, pending_offsets_v = [[[]]], [[[]]]
    # HB doesn't include the root node
    selected_rects_l, selected_rects_u = [], [] 
    while len(pending_l) != 0:
        cur_rect_l = pending_l.pop()
        cur_rect_u = pending_u.pop()
        offset_h = pending_offsets_h.pop()
        offset_v = pending_offsets_v.pop()
        # resolve offsets in any pending ranges
        lower, higher = expand_offsets(cur_rect_l, cur_rect_u, offset_h, offset_v)
        selected_rects_l.extend(lower)
        selected_rects_u.extend(higher)

        sub_rects_l, sub_rects_u = split_rectangle((cur_rect_l, cur_rect_u), b_h, b_v)
        same_shape_groups = same_shape(sub_rects_l, sub_rects_u)

        for group_idx in same_shape_groups:
            sub_rects_l_group = np.array(sub_rects_l)[group_idx]
            sub_rects_u_group = np.array(sub_rects_u)[group_idx]
            # If it's a leaf don't add to pending rects,
            # append Identity at the end of selection.
            if (sub_rects_u_group[0][0] - sub_rects_l_group[0][0] + 1) * \
               (sub_rects_u_group[0][1] - sub_rects_l_group[0][1] + 1) > 1:

                offset_h_level, offset_v_level = [], []
                for l,u in sub_rects_l_group:
                    offset_h_level.append(l - sub_rects_l_group[0][0])
                    offset_v_level.append(u - sub_rects_l_group[0][1])
                # append the first quad in subgroup to pending list
                pending_l.append(sub_rects_l_group[0])
                pending_u.append(sub_rects_u_group[0])
                pending_offsets_h.append(offset_h + [offset_h_level])
                pending_offsets_v.append(offset_v + [offset_v_level])

    lower = np.array(selected_rects_l, dtype=np.int32)
    upper = np.array(selected_rects_u, dtype=np.int32)
    M = workload.RangeQueries((n,m), lower, upper, np.float32)

    return matrix.VStack([M, matrix.Identity(n*m)])

def GenerateCells(n,m,num1,num2,grid):
    # this function used to generate all the cells in UGrid
    assert math.ceil(util.old_div(n,float(grid))) == num1 and math.ceil(util.old_div(m,float(grid))) == num2, "Unable to generate cells for Ugrid: check grid number and grid size"
    lower, upper = [], []
    for i in range(num1):
        for j in range(num2):
            lb = [int(i*grid),int(j*grid)]
            rb = [int((i+1)*grid-1),int((j+1)*grid-1)]
            if rb[0] >= n:
                rb[0] = int(n-1)
            if rb[1] >= m:
                rb[1] = int(m-1)

            lower.append(lb)
            upper.append(rb)

    return lower, upper


class Identity(SelectionOperator):

    def __init__(self, domain_shape):
        super(Identity, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, "Identity selection only workss for 1D"
        self.domain_shape = domain_shape

    def select(self):
        return matrix.Identity(self.domain_shape[0])
        #return sparse.identity(self.domain_shape[0])


class Total(SelectionOperator):

    def __init__(self, domain_shape):
        super(Total, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, "Total selection only works for 1D"
        self.domain_shape = domain_shape

    def select(self):
        return workload.Total(self.domain_shape[0])
        #return np.ones((1, self.domain_shape[0]), dtype=np.float)


class H2(SelectionOperator):
    """
    H2 select operator(1D): 
    Adds hierarchical queries with uniform branching factor.
    Works in an iterative top-down fashion splits each node V to create max{b, |V|} children and stops
    when a node has 1 element. This creates a balanced tree with maximum number of nodes at each level
    except (possibly) the last one.

    """

    def __init__(self, domain_shape, branching=2):
        super(H2, self).__init__()

        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 1, 'Hierarchical selection only supports 1D and 2D domain shapes'
        assert branching > 1
        self.branching = branching
        self.domain_shape = domain_shape
        self.cache = np.ones((1, 1))

    def select(self):
        if self.domain_shape[0] == self.cache.shape[1]:
            h_queries = self.cache
        else:
            h_queries = buildHierarchical_sparse(self.domain_shape[0], self.branching)

        return matrix.EkteloMatrix(h_queries)


class HB(SelectionOperator):
    '''
    HB select operator(1D and 2D)
    Add hierarchical queries with optimal branching factor, per Qardaji et al.
    '''

    def __init__(self, domain_shape):
        super(HB, self).__init__()

        assert (isinstance(domain_shape, tuple) and len(domain_shape) == 1
                or len(domain_shape) == 2
                ), 'HB selection only supports 1D and 2D domain shapes'
        self.domain_shape = domain_shape

    def select(self):
        if len(self.domain_shape) == 1:

            N = self.domain_shape[0]
            branching = find_best_branching(N)
            # remove root
            h_queries = buildHierarchical_sparse(N, branching).tocsr()[1:]
            h_queries = matrix.EkteloMatrix(h_queries)

        elif len(self.domain_shape) == 2:
            N = self.domain_shape[0] * self.domain_shape[1]
            branching = find_best_branching(N)
            h_queries = Hb2D(self.domain_shape[0], 
                             self.domain_shape[1],
                             branching, 
                             branching)
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
        QtQ = self.W.gram().dense_matrix()
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

        return matrix.EkteloMatrix(mat)

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

    def __init__(self, domain_shape):
        super(QuadTree, self).__init__()
        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 2, "QuadTree selection only workss for 2D"
        self.domain_shape = domain_shape

    @staticmethod
    def rect_to_quads(rect_range):
        '''
        Given an rectangular domain represented using boarder cordinates (upper_left, lower_right),
        it splits it correctly to 4 quads in the midpoints
        '''
        ul, lr = rect_range
        upper, left =  ul
        lower, right = lr   

        n_rows = lower - upper + 1
        n_cols = right - left + 1   

        # If ncol is odd, do vert splits in balanced manner
        col_parity = 0
        if n_cols % 2:
            col_parity = 1
        col_midpoint = left + util.old_div(n_cols, 2)
        row_midpoint = upper + util.old_div(n_rows, 2)  

        if n_rows == 1:
            # if x has only one row then do only vertical split
            row = lr[0]
            return [ul, (row, col_midpoint)], [(row, col_midpoint - 1), lr]

        if n_cols == 1:
            # if x has only one col then do only horizontal split
            col = lr[1]
            return [ul, (row_midpoint, col)] , [(row_midpoint - 1, col), lr]

        # o/w do both splits
        q1 = (ul,                                        (row_midpoint - 1, col_midpoint - 1))
        q2 = ((upper, col_midpoint),                     (row_midpoint - 1, right) )
        q3 = ((row_midpoint, left),                      (lower, col_midpoint - 1 + col_parity))
        q4 = ((row_midpoint, col_midpoint + col_parity), lr)

        lower = [coordinates[0] for coordinates in [q1, q2, q3, q4] ]
        upper = [coordinates[1] for coordinates in [q1, q2, q3, q4] ]

        return lower, upper


    @staticmethod
    def quadtree(n, m):
        ''' Quadtree function, accepts a shape (n, m)
            and returns a set of measurements in sparse format on the expanded x vector
            n and m can be arbitrary numbers
        '''
        # Avoid doing recursion, python is notoriously bad at it
        pending_l, pending_u = [ (0, 0) ], [(n - 1, m - 1)]
        # Avoid calculating partition for quads of the same shape
        # by storing a base quad and a list of offset in each level of the hierarchy 
        pending_offsets_h, pending_offsets_v = [[[0]]], [[[0]]]
        selected_rects_l, selected_rects_u = [], [] 
        while len(pending_l) != 0:
            cur_rect_l = pending_l.pop()
            cur_rect_u = pending_u.pop()
            offset_h = pending_offsets_h.pop()
            offset_v = pending_offsets_v.pop()
            # resolve offsets in any pending ranges
            lower, higher = expand_offsets(cur_rect_l, cur_rect_u, offset_h, offset_v)
            selected_rects_l.extend(lower)
            selected_rects_u.extend(higher)

            sub_rects_l, sub_rects_u = QuadTree.rect_to_quads((cur_rect_l, cur_rect_u))
            same_shape_groups = same_shape(sub_rects_l, sub_rects_u)

            for group_idx in same_shape_groups:
                sub_rects_l_group = np.array(sub_rects_l)[group_idx]
                sub_rects_u_group = np.array(sub_rects_u)[group_idx]
                # If it's a leaf don't add to pending rects,
                # append Identity at the end of selection.
                if (sub_rects_u_group[0][0] - sub_rects_l_group[0][0] + 1) * \
                   (sub_rects_u_group[0][1] - sub_rects_l_group[0][1] + 1) > 1:

                    offset_h_level, offset_v_level = [], []
                    for l,u in sub_rects_l_group:
                        offset_h_level.append(l - sub_rects_l_group[0][0])
                        offset_v_level.append(u - sub_rects_l_group[0][1])
                    # append the first quad in subgroup to pending list
                    pending_l.append(sub_rects_l_group[0])
                    pending_u.append(sub_rects_u_group[0])
                    pending_offsets_h.append(offset_h + [offset_h_level])
                    pending_offsets_v.append(offset_v + [offset_v_level])
    
        lower = np.array(selected_rects_l, dtype=np.int32)
        upper = np.array(selected_rects_u, dtype=np.int32)
        M = workload.RangeQueries((n,m), lower, upper, np.float32)

        return matrix.VStack([M, matrix.Identity(n*m)])

    def select(self):
        return QuadTree.quadtree(self.domain_shape[0], self.domain_shape[1])

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

        lower, upper = GenerateCells(n, m, num1, num2, grid)
        return workload.RangeQueries((n, m), np.array(lower), np.array(upper))


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
            return workload.RangeQueries((1,1), lower=np.array([[0,0]]), higher=np.array([[0,0]]))

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
            lower, higher = split_rectangle(((0,0), (nn - 1, mm - 1)), num1, num2)
            
        return workload.RangeQueries(self.domain_shape, np.array(lower), np.array(higher))


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

        return matrix.EkteloMatrix(wavelet_query)


class AddEquiWidthIntervals(SelectionOperator):
    """
    Given a selected query, the first row in W, it complements select query with a set of disjoint interval queries, of
    a specified width.
    """
    def __init__(self, W, log_width):
        super(AddEquiWidthIntervals, self).__init__()
        #self.M_hat = support.extract_M(W)   # expects W to contain a single measurement
        self.M_hat = W.sparse_matrix()
        self.grid_size = min(2 ** log_width, self.M_hat.shape[1])

    def select(self):
        mat = sparse.vstack((self.M_hat, support.complement(self.M_hat, self.grid_size)))
        return matrix.EkteloMatrix(mat)

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

