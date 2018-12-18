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
from ektelo import matrix, workload, hdmm_templates
from functools import reduce


def GenerateCells(n,m,num1,num2,grid):
    '''
    Generate grid shaped celles for UniformGrid and AdaptiveGrid.and
    n, m: 2D domain shape
    num1, num2: number of cells along two dimensions
    grid: grid size
    '''
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

class HierarchicalRanges(SelectionOperator):
    ''' 
    ND hiearchical selection operator
    At any level of the Hiearchy, the domain is partitioned using some
    data-independent function. This class provide efficient implementation
    to generate the lower and upper boudaries through the ranges_gen() function.
    '''

    def __init__(self, domain_shape, include_root, fast_leaves, split_func, **kwargs):
        self.domain_shape = domain_shape
        self.include_root = include_root
        self.fast_leaves = fast_leaves
        self.split_func = split_func
        self.kwargs = kwargs


    @staticmethod    
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

    @staticmethod
    def quick_product(*arrays):
        '''
        Quick calculation of cross products of a list of arrays.
        Return the sum of each result
        '''
        la = len(arrays)
        arr = np.empty([la] + [len(a) for a in arrays], dtype='int')
        for i, a in enumerate(np.ix_(*arrays)):
            arr[i, ...] = a
        return arr.reshape(la, -1)

    @staticmethod
    def expand_offsets(cur_rect_l, cur_rect_u, offsets):
        '''
        Expand offsets at different level along each dimension to generate the 
        final offsets for all candidate by computing the sum of each tuple in the 
        cross product of offset arrays.
        e.g For the some dimension two level offsets [[0, 1, 0], [2, 4, 2]] will be expanded to 
        [2 4 2 3 5 3 2 4 2]
        cur_rect_l and cur_rect_u: coordinates of the lower and upper corner of the range.
        offsets: Nested array representing offsets of ranges along dimension, level of hierarchy    

        ''' 
        # remove empty list(no query at this level)
        offsets = [list(filter(lambda x: len(x) > 0, d)) for d in offsets]
        assert all([len(d) == len(offsets[0]) for d in offsets]),\
               "Shape of offsets along each dimension should match."    
        if len(offsets[0]) < 1:
            return [], []   
        # expand offsets across different levels.
        expanded_offsets = [HierarchicalRanges.quick_product(*d).sum(axis=0) for d in offsets] 
        lower = np.vstack([ l + offset for l, offset in zip(cur_rect_l, expanded_offsets)]).T
        upper = np.vstack([ u + offset for u, offset in zip(cur_rect_u, expanded_offsets)]).T
        return lower, upper

    @staticmethod
    def grid_split_range(cur_range_l, cur_range_u, **kwargs):
        """
        Split ND-range into grids according to branching factors along each dimension
        cur_range_l, cur_range_u: coordinates of the lower and upper boundary 
        kwargs: needs to have a 'branching_list' memeber with the branching factor along each dimension
        """
        branchings = kwargs['branching_list']
        dim_lens = np.array(cur_range_u) - np.array(cur_range_l) + 1
        assert len(branchings) == len(dim_lens), "The numbers of dimension and branching factors need to match"
        def get_boarder(dim_len, branching):
            if branching > dim_len:
                split_num = dim_len
                boarder = [ (i, i) for i in range(split_num)]
            elif dim_len % branching != 0:
                new_hsize = np.divide(float(dim_len), branching)
                split_num = [np.ceil(new_hsize * (i + 1)).astype(int) for i in range(branching - 1)]
                temp = [i -1 for i in split_num]
                boarder = list(zip(([0] + split_num), (temp + [dim_len-1])) )
            else:
                cell_size_h = util.old_div(dim_len, branching)
                boarder = [(i * cell_size_h, (i+1) * cell_size_h - 1) for i in range(branching)]
            return boarder
        # get back boarder along each dimension
        boarder_list = [get_boarder(d, b) for d, b in zip(dim_lens, branchings)]

        try:
            # use quick_product to calculate crossproduct if all dimensions breaks into the same shape
            lower, upper = np.array(boarder_list).transpose([2,0,1])
            lower_list = HierarchicalRanges.quick_product(*lower).T
            upper_list = HierarchicalRanges.quick_product(*upper).T
        except ValueError:
            # else fall back to standard crossproduct, will be slower when results size get large
            x = np.array(list(itertools.product(*boarder_list)))
            lower_list, upper_list = x.transpose([2,0,1])
        
        lower_list = lower_list + cur_range_l
        upper_list = upper_list + cur_range_l
        return lower_list, upper_list

    @staticmethod
    def ranges_gen(start_range_l, start_range_u, include_root, include_leaf, split_func, **kwargs):
        '''
        Fast Hierarchical range generation method by iteratively partitioning the domain and 
        resusing subpartition of the same shape. At each level, only one canonical range is stored for all 
        ranges of the same shape, and any range is represented as a offset. 
        
        start_range_l, start_range_u: lower and upper boundaries of top level range to start with
        include_root: include the root node(start_range) if set to True 
        include_leave: include the leaf nodes(cells of size 1) if set to True
        split_func: spliting function at each level, takes parameter (cur_range_l, cur_range_u, **kwargs)
                    cur_range_l, cur_range_u are the lower and upper boundardy of the range to be split.
        **kwargs: argument list, will be passed into split_func
        Return: list of lower and upper bound

        '''
        # Avoid doing recursion, python is notoriously bad at it
        assert len(start_range_l) == len(start_range_u), \
                "Dimesions of lower and upper boundaries shoud match"
        pending_l, pending_u = [start_range_l], [start_range_u]
        dimension = len(start_range_l)
        # List of pending offsets: dimension, level, duplicates within level.
        pending_offsets = [[[np.array([0], dtype='int')]] * dimension] if include_root \
                        else [[[np.array([], dtype='int')]] * dimension]
        selected_ranges_l, selected_ranges_u = [], [] 
        
        while len(pending_l) != 0:
            cur_range_l = pending_l.pop()
            cur_range_u = pending_u.pop()
            cur_offset = pending_offsets.pop()
            # Resolve offsets in any pending ranges
            lower, higher = HierarchicalRanges.expand_offsets(cur_range_l, cur_range_u, cur_offset)
            selected_ranges_l.extend(lower)
            selected_ranges_u.extend(higher) 
            sub_ranges_l, sub_ranges_u = split_func(cur_range_l, cur_range_u, **kwargs)
            # Put subranges of same shapes into groups, select the first subrect as canonical 
            # and represent the rest using offsets. 
            same_shape_groups = HierarchicalRanges.same_shape(sub_ranges_l, sub_ranges_u)
            for group_idx in same_shape_groups:
                sub_ranges_l_group = np.array(sub_ranges_l)[group_idx]
                sub_ranges_u_group = np.array(sub_ranges_u)[group_idx]
                # Select the first range as the canonical range of the group
                canonical_range_l, canonical_range_u = sub_ranges_l_group[0], sub_ranges_u_group[0]
                canonical_shape = [ u-l+1 for l,u in zip(canonical_range_l, canonical_range_u)]
                # If it's a leaf don't add to pending rects,
                # append Identity at the end of selection.
                if np.prod(canonical_shape) > 1:
                    new_offset = []
                    for d in range(len(cur_offset)):
                        offset_level = np.array(sub_ranges_l_group)[:, d] - canonical_range_l[d]
                        # augment offset list by one level for each dimension
                        new_offset.append(cur_offset[d] + [offset_level])
                    # append the first quad in subgroup to pending list
                    pending_l.append(sub_ranges_l_group[0])
                    pending_u.append(sub_ranges_u_group[0])
                    pending_offsets.append(new_offset)
                elif include_leaf:
                    # if include_leaf is set, then add to selected
                    for l, u in zip(sub_ranges_l_group, sub_ranges_u_group):
                        lower, higher = HierarchicalRanges.expand_offsets(l, u, cur_offset)
                        selected_ranges_l.extend(lower)
                        selected_ranges_u.extend(higher)

        lower = np.array(selected_ranges_l, dtype=np.int32)
        upper = np.array(selected_ranges_u, dtype=np.int32)
        return lower, upper

    def select(self):
        dimension = len(self.domain_shape)
        start_range_l = [0] * dimension
        start_range_u = [d - 1 for d in self.domain_shape]

        # Optimization using Identity as leaves measurements
        if self.fast_leaves:
            lower, upper = HierarchicalRanges.ranges_gen(
                                                    start_range_l, 
                                                    start_range_u,
                                                    self.include_root, 
                                                    False, 
                                                    self.split_func, 
                                                    **self.kwargs)

            # If all selected queries are leaves, return Identity directly
            if len(lower) == 0:
                return  matrix.Identity(np.prod(self.domain_shape))
            # Otherwise union M with Identity
            M = workload.RangeQueries(self.domain_shape, lower, upper, np.float32)
            return matrix.VStack([M, matrix.Identity(np.prod(self.domain_shape))])

        else:
            lower, upper = HierarchicalRanges.ranges_gen(
                                                    start_range_l, 
                                                    start_range_u,
                                                    self.include_root, 
                                                    True, 
                                                    self.split_func, 
                                                    **self.kwargs)

            # directly return results with leaves included
            M = workload.RangeQueries(self.domain_shape, lower, upper, np.float32)
            return M


class H2(HierarchicalRanges):
    ''' 
    H2 select operator(ND): 
    Adds hierarchical queries with uniform branching factor.
    Works in an iterative top-down fashion splits each node V to create max{b, |V|} children and stops
    when a node has 1 element. This creates a balanced tree with maximum number of nodes at each level
    except (possibly) the last one.
    '''
    def __init__(self, domain_shape, branching=2, fast_leaves=True):

        dimension = len(domain_shape)
        super(H2, self).__init__(domain_shape=domain_shape,
                                 include_root=True,
                                 fast_leaves=fast_leaves,
                                 split_func=H2.grid_split_range,
                                 branching_list=[branching]*dimension)



class HB(HierarchicalRanges):
    '''
    HB select operator(ND)
    Add hierarchical queries with optimal branching factor, per Qardaji et al.
    '''
    def __init__(self, domain_shape, include_root=False, fast_leaves=True):

        dimension = len(domain_shape)
        branching = HB.find_best_branching(np.prod(domain_shape))

        super(HB, self).__init__(domain_shape=domain_shape,
                                 include_root=include_root,
                                 fast_leaves=fast_leaves,
                                 split_func=HB.grid_split_range,
                                 branching_list=[branching]*dimension)

    @staticmethod
    def variance(N, b):
        '''Computes variance given domain of size N
        and branchng factor b.  Equation 3 from paper.'''
        h = math.ceil(math.log(N, b))
        return ( ((b - 1) * h**3) - (util.old_div((2 * (b+1) * h**2), 3)))

    @staticmethod
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
            v = HB.variance(N, b)
            if v < min_v:
                min_v = v
                min_b = b
        return min_b


class QuadTree(HierarchicalRanges):
    '''
    QuadTree Selection operator

    '''
    def __init__(self, domain_shape, fast_leaves=True):
        assert len(domain_shape) == 2, \
            "QuadTree selection is defined on 2D"
        domain_shape = domain_shape

        super(QuadTree, self).__init__(domain_shape=domain_shape,
                                       include_root=True,
                                       fast_leaves=fast_leaves,
                                       split_func=QuadTree.quad_split_range)

    @staticmethod
    def quad_split_range(cur_range_l, cur_range_u, **kwarg):
        '''
        Given an rectangular domain represented using boarder cordinates (upper_left, lower_right),
        it splits it correctly to 4 quads in the midpoints
        '''
        ul, lr = cur_range_l, cur_range_u
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

class HDMM1D(SelectionOperator):
    def __init__(self, W, p=None):
        self.W = W
        self.n = W.shape[1]
        self.p = p
        if p is None:
            self.p = self.n // 16
    
    def select(self):
        pid = hdmm_templates.PIdentity(self.p, self.n)
        pid.optimize(self.W)
        return pid.strategy()

class HDMM(SelectionOperator):
    def __init__(self, domain_shape, W, ps):
        """ only works for kronecker product or union of kron W
        ps is hyperparameter to HDMM, must be same length as # dimensions

        Example Usage:
        P = workload.Prefix(32)
        W = workload.Kronecker([P, P])
        sel = selection.HDMM([32,32], W, [2,2])
        M = sel.select()
        """
        self.W = W 
        self.domain_shape = domain_shape
        self.template = hdmm_templates.KronPIdentity(ps, domain_shape)

    def select(self):
        self.template.optimize(self.W)
        return self.template.strategy()

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


class AdaptiveGrid(HierarchicalRanges):

    def __init__(self, domain_shape, x_hat, eps_par, c2=5):
        assert isinstance(domain_shape, tuple) and len(
            domain_shape) == 2, "AdptiveGrid selection only works for 2D"

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
            lower, higher = AdaptiveGrid.grid_split_range( (0,0), (nn - 1, mm - 1) , branching_list=[num1, num2])
            
        return workload.RangeQueries(self.domain_shape, np.array(lower), np.array(higher))

class HD_IHB(SelectionOperator):
    '''
    fast global selection for HB_STRIPED
    '''
    def __init__(self, domain_shape, impl='MM', hb_dim=-1):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.impl = impl
        self.hb_dim = hb_dim # default to last dimension

    def select(self):
        N = self.domain_shape[self.hb_dim]
        domains = list(self.domain_shape)
        del domains[self.hb_dim]

        if self.impl == 'MM':
            I = matrix.Identity(int(np.prod(domains)))
            H = HB((N,)).select()
        elif self.impl == 'sparse':
            I = matrix.Identity(int(np.prod(domains))).sparse_matrix()
            H = HB((N,)).select().sparse_matrix()
        elif self.impl == 'dense':
            I = matrix.Identity(int(np.prod(domains))).dense_matrix()
            H = HB((N,)).select().dense_matrix()
        else:
            print("Invalid measurement type", self.impl)
            exit(1)
        return matrix.Kronecker([I, H])

class Wavelet(SelectionOperator):
    '''
    Adds wavelet matrix as measurements
    '''

    def __init__(self, domain_shape):
        super(Wavelet, self).__init__()

        self.domain_shape = domain_shape
        assert all(n & (n-1) == 0 for n in domain_shape),\
                'each dimension of domain must be a power of 2'

    def select(self):
        if len(self.domain_shape) == 1:
            return matrix.Haar(self.domain_shape[0])
        return matrix.Kronecker([matrix.Haar(n) for n in self.domain_shape])

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

