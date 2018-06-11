from ektelo import support
from ektelo import util
from ektelo.operators import MapperOperator
from ektelo.algorithm.ahp import ahp_fast
from functools import reduce
import numpy as np
import math


def general_pairing(tup):
    """ Generalize cantor pairing to k dimensions """
    if len(tup) == 0:
        return tup[0]  # no need for pairing in 1D case
    else:
        return reduce(support.cantor_pairing, tup)


def partition_grid(domain_shape, grid_shape, canonical_order=False):
    """
    :param domain_shape: a shape tuple describing the domain, e.g (6,6) (in 2D)
    :param grid_shape: a shape tuple describing cells to be grouped, e.g. (2,3) to form groups of 2 rows and 3 cols
        note: in 1D both of the above params can simply be integers
    :return: a partition array in which grouped cells are assigned some unique 'group id' values
             no guarantee on order of the group ids, only that they are unique
    """

    # allow for integers instead of shape tuples in 1D
    if isinstance(domain_shape, int):
        domain_shape = (domain_shape, )
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,)

    assert sum(divmod(d,b)[1] for (d,b) in zip(domain_shape, grid_shape)) == 0, "Domain size along each dimension should be a multiple of size of block"

    def g(*idx):
        """
        This function will receive an index tuple from numpy.fromfunction
        It's behavior depends on grid_shape: take (i,j) and divide by grid_shape (in each dimension)
        That becomes an identifier of the block; then assign a unique integer to it using pairing.
        """
        x = np.array(idx)
        y = np.array(grid_shape)

        return general_pairing( util.old_div(x,y) )  # broadcasting integer division

    h = np.vectorize(g)

    # numpy.fromfunction builds an array of domain_shape by calling a function with each index tuple (e.g. (i,j))
    partition_array = np.fromfunction(h, domain_shape, dtype=int)

    if canonical_order:
        partition_array = support.canonical_ordering(partition_array)

    return partition_array

def cells_to_mapping(cells, domain):

    n,m = domain
    partition_vector = np.empty([n,m],dtype= int)
    group_no = 0
    for ul,lr in cells:
        up, left = ul; low, right = lr
        for row in range(up,low+1):
            partition_vector[row,left:right+1] = group_no
       
        group_no+=1
    return support.canonical_ordering(partition_vector).flatten()


class Grid(MapperOperator):

    stability = 1

    def __init__(self, orig_shape, grid_shape, canonical_order=True):
        super(Grid, self).__init__()

        self.orig_shape = orig_shape
        self.grid_shape = grid_shape
        self.canonical_order = canonical_order

    def mapping(self):
        return partition_grid(self.orig_shape, self.grid_shape, self.canonical_order)


class Striped(MapperOperator):
    """
    create a striped partition object, which groups all cells along stripe_dim together

    :param domain_size: the size of the domain
    :param stripe_dim: the dimension to apply the stripe on
    :return: A partition object
    """
    stability = 1

    def __init__(self, domain_size, stripe_dim):
        super(Striped, self).__init__()
        
        self.domain_size = domain_size
        self.stripe_dim = stripe_dim

    def mapping(self):
        vectors = [np.arange(dom, dtype=int) for dom in self.domain_size]
        vectors[self.stripe_dim] = np.ones(self.domain_size[self.stripe_dim], dtype=int)

        return support.combine_all(vectors).flatten()


class HilbertTransform(MapperOperator):
    '''
    Transform 2D domain to 1D domain according to hilbert curve. 
    Usesd in 2D DAWA as preprocessing. 
    The current implementation requires the domain to be a square of length which is a power of 2
    '''
    stability = 1

    def __init__(self, domain_shape):
        super(HilbertTransform, self).__init__()
        
        self.domain_shape = domain_shape

    @staticmethod
    def hilbert(N):
        """
        Produce coordinates of an NxN Hilbert curve.    

        @param N:
             the length of side, assumed to be a power of 2 ( >= 2) 

        @returns:
              x and y, each as an array of integers representing coordinates
              of points along the Hilbert curve. Calling plot(x, y)
              will plot the Hilbert curve.  

        From Wikipedia
        """
        assert 2**int(math.ceil(math.log(N, 2))) == N, "N={0} is not a power of 2!".format(N)
        if N==2:
            return  np.array((0, 0, 1, 1)), np.array((0, 1, 1, 0))
        else:
            x, y = HilbertTransform.hilbert(util.old_div(N,2))
            xl = np.r_[y, x,     util.old_div(N,2)+x, N-1-y  ]
            yl = np.r_[x, util.old_div(N,2)+y, util.old_div(N,2)+y, util.old_div(N,2)-1-x]
            return xl, yl

    def mapping(self):
        assert len(self.domain_shape) == 2, "Only supports 2D Hilbert transform"
        shape1, shape2 = self.domain_shape
        assert shape1 == shape2 and 2**int(math.ceil(math.log(shape1, 2))) == shape2

        x,y = HilbertTransform.hilbert(shape1)
        partition_vec = np.zeros_like(x)

        index = np.ravel_multi_index([x,y], (shape1,shape2))
        partition_vec[index] = list(range(len(x)))


        return partition_vec


class AHPCluster(MapperOperator):
    """
    Calculation ahp cluster based on noisy estimation xest
    """

    stability = 1

    def __init__(self, xest, eps_par):
        super(AHPCluster, self).__init__()

        self.xest = xest
        self.eps_par = eps_par

    def get_rank(self):
        """
        helper function to get an array that represents the rank of each item in the xest
        e.g. [1,2,3] => [0,1,2];[5,4,3,2,1] =>[4,3,2,1,0] ; [4,3,1,5,2] => [3 2 0 4 1] 
        """
        order = np.argsort(self.xest)
        rank = np.empty(len(self.xest),int)
        rank[order]= np.arange(len(self.xest))
        return rank

    def mapping(self):
        # sort
        rank = self.get_rank()
        sorted_xest = np.sort(self.xest)

        # clustering
        cluster = ahp_fast.greedy_cluster(sorted_xest,self.eps_par)
        partition_vec = support.get_partition_vec(rank,len(sorted_xest), cluster, closeRange=False)

        return partition_vec

class UGridPartition(MapperOperator):

    def __init__(self, domain_shape, data_sum, eps_par, ag_flag=False, c = 10, gz =0):
        super(UGridPartition, self).__init__()
        # this needs an estimate of the epsilon used to estimate the data
        self.c = c
        self.gz = gz
        self.domain_shape = domain_shape
        self.data_sum = data_sum
        self.eps_par = eps_par
        # when ug is used as the first level of ag, calculation of grid size 
        # is slightly different. 
        self.ag_flag = ag_flag 

    @staticmethod
    def GenerateCells(n,m,num1,num2,grid):
        # this function used to generate all the cells in UGrid
        assert math.ceil(n/float(grid)) == num1 and math.ceil(m/float(grid)) == num2, "Unable to generate cells for Ugrid: check grid number and grid size" 
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
    
    def mapping(self):
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
        cells = UGridPartition.GenerateCells(n, m, num1, num2, grid)
        return cells_to_mapping(cells, (n,m))


class WorkloadBased(MapperOperator):

    stability = 1

    def __init__(self, W, canonical_order=True):
        super(WorkloadBased, self).__init__()

        self.W = W
        self.canonical_order = canonical_order

    @staticmethod
    def partition_lossless(A, stable=True):
        """
        :param A:  Matrix (either sparse or dense) whose columns will be analyzed to produce a partition (typically workload matrix or measurement matrix)
                        (queries will be rows here, even if they are flattened k-dimensional queries)
        :return:   1D partition vector encoding groups
        The returned partition describes groups derived from matching columns (i.e. columns that are component-wise equal)
        (There is no guarantee that new ordering of groups will be "stable" w.r.t original ordering of columns)
        """
        # mat = A.T # transpose so we can operate on rows
        # form a view where each row is joined into void type
        # for explanation, see: http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        # (couldn't make this work on columns)
        # compressed = numpy.ascontiguousarray(mat).view(numpy.dtype((numpy.void, mat.dtype.itemsize * mat.shape[1])))  

        # http://www.ryanhmckenna.com/2017/01/efficiently-remove-duplicate-rows-from.html
        v = numpy.random.rand(A.shape[0])
        vA = A.T.dot(v)     

        # use numpy.unique
        # returned 'inverse' is the index of the unique value present in each position (this functions as a group id)
        _u, index, inverse = numpy.unique(vA, return_index=True, return_inverse=True)   

        if stable:
            return support.canonical_ordering(_replace(inverse, index), canonical_order=True)
        else:
            return support.canonical_ordering( inverse, canonical_order=True)
    
    def mapping(self):
        return WorkloadBased.partition_lossless(self.W)

