from __future__ import division
from builtins import zip
from ektelo import util
from functools import reduce
import math
import numpy as np
from scipy import sparse


def split_rectangle (x, b_h, b_v):
    """
    Check if the quadtree produces a measurement set of correct size
    For use with Hb2D
    """
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    # if equally divisible then b_{v,h} is the number of split points for each dimension
    h_split = b_h
    v_split = b_v

    # otherwise create the list of splitpoints
    if n_rows % b_h != 0:
        new_hsize = np.divide(n_rows, b_h, dtype = float)
        h_split = [np.ceil(new_hsize * (i + 1)).astype(int) for i in range(b_h - 1)]

    if n_cols % b_v != 0:
        new_vsize = np.divide(n_cols, b_v, dtype = float)
        v_split = [np.ceil(new_vsize * (i + 1)).astype(int) for i in range(b_v - 1)]

    if b_h > n_rows:
        h_split = n_rows
    if b_v > n_cols:
        v_split = n_cols
    # if x has only one row then do only vertical split
    if x.shape[0] == 1:
        final_rects = np.split(x, v_split, axis = 1)

        return final_rects

    # if x has only one col then do only horizontal split
    if x.shape[1] == 1:
        final_rects = np.split(x, h_split, axis = 0)

        return final_rects

    # o/w do both splits
    final_rects = []
    h_rects = np.split(x, h_split, axis = 0)
    for h_rect in h_rects:
        v_rects = np.split(h_rect,  v_split, axis = 1)
        for v_rect in v_rects:
            final_rects.append(v_rect)

    return final_rects


def cantor_pairing(a, b):
    """
    A function returning a unique positive integer for every pair (a,b) of positive integers
    """
    return (a+b)*(a+b+1)/2 + b
    
def _replace(vector, new_values):
    for i in range(len(vector)):
        vector[i] = new_values[ vector[i] ]
    return vector

def get_partition_vec(rank,n,cluster,closeRange=False):
    """ get the partition vector from clusters returned by partition algorithms
        rank: If the bins are sorted, give the rank of each item in the input list.
            Used by AHP partition. Set rank = None if not used.
        n: Length of vector in original domain
        cluster: Cluster/partition returned by partition algorithms
        closeRange: if set to True, ranges in clusters are close range. (DAWA partition)
            i.e. [a,b] indicates [a,a+1,...b-1,b]
            if set to False, ranges in clusters are default python representation. (AHP partition)
            i.e. [a,b] indicates [a,a+1,...b-1]
    """
    partition_vec_sorted = np.empty(n,int)
    assert cluster[0][0] == 0,"First bin of partition must start with 0"
    # assign groupID to elements in sorted list.
    for i in range(len(cluster)):
        if closeRange:
            assert cluster[-1][1] == n-1, " Last bin of partition must end with length of original data"
            partition_vec_sorted[cluster[i][0]:cluster[i][1]+1] = i
        else:
            assert cluster[-1][1] == n, " Last bin of partition must end with length of original data"
            partition_vec_sorted[cluster[i][0]:cluster[i][1]] = i
    # get index in sorted list for elements in original domain, then get groupID.
    if rank is None:
        partition_vec = partition_vec_sorted
    else:
        partition_vec = np.array([partition_vec_sorted[rank[i]] for i in range(n)] )

    return partition_vec


def update_corners(corner, groupID, row, start, end):
    ''' helper function for get_subdomain
        update corners coordinates for a certain group.
        return False if the domain is not rectangular
    '''
    # if it is the first ocurrence of the group
    # update the upper left and upper right corner
    if groupID not in corner:
        corner[groupID] = {'ul':(row, start),'ur':(row,end), 'll':(row, start),'lr':(row,end)}
    else:
        temp = corner[groupID]
        if row == temp['ll'][0]: # incontinous group on the upper line
            return False 

        # update the lower corners
        # make sure the columns match and rows are continous.
        if temp['ll'][1] == start and temp['lr'][1] == end and temp['ll'][0] == row-1:
            # move the lower corners one line lower
            corner[groupID]['ll'] = (temp['ll'][0]+1, temp['ll'][1])
            corner[groupID]['lr'] = (temp['lr'][0]+1, temp['lr'][1])
        else:
            return False

    return True



def get_subdomain_grid(mapping, domain_shape):
    '''
    Given a mapping, return the domain size of all the subdomain when it is 
    used by the SplitByPartition operator.
    The original domain needs to be 2D and the mapping should split the domain 
    to smaller grids. Non-rectangular subdomain shapes are not supported,
    None will be returned.

    '''
    assert len(domain_shape) == 2 , 'Only works for 2D domains'
    m, n = domain_shape
    # unflatten the mapping vector
    mapping = mapping.reshape(domain_shape)
    corners = {}
    # record corners of each group in one pass of the mapping vector
    for i in range(m):
        start = 0
        for j in range(n):
            if j+1 >= n or mapping[i][j] != mapping[i][j+1]:
                groupID = mapping[i][start]
                status = update_corners(corners, groupID, i, start, j)
                start = j+1
                if status == False:
                    return None

    # calculate subdomains from corners
    sub_domains = {}
    for g in corners:
        temp = corners[g]
        sub_domains[g] = (temp['ll'][0] - temp['ul'][0] + 1, temp['ur'][1] - temp['ul'][1] + 1)

    return sub_domains


def canonical_ordering(mapping):
    """ remap according to the canonical order.
     if bins are noncontiguous, use position of first occurrence.
     e.g. [3,4,1,1] => [1,2,3,3]; [3,4,1,1,0,1]=>[0,1,2,2,3,2]
    """
    unique, indices, inverse, counts = mapping_statistics(mapping)

    uniqueInverse, indexInverse = np.unique(inverse,return_index =True)

    indexInverse.sort()
    newIndex = inverse[indexInverse]
    tups = list(zip(uniqueInverse, newIndex)) 
    tups.sort(key=lambda x: x[1])
    u = np.array( [u for (u,i) in tups] )
    mapping = u[inverse].reshape(mapping.shape)

    return mapping


def mapping_statistics(mapping):
    return np.unique(mapping, return_index=True, return_inverse=True, return_counts=True)   

##############################################################
# Transformation helpers
##############################################################

def project(mapping, idx, vector):
    return projection_matrix(mapping, idx) * vector

def unproject(mapping, idx, vector):
    return vector * projection_matrix(mapping, idx)

def reduce_data(mapping, data):
    return expansion_matrix(mapping) * data

def expand_data(mapping, data):
    return reduction_matrix(mapping) * data

def reduce_queries(mapping, queries):
    return query * expansion_matrix(mapping) 

def expand_queries(mapping, queries):
    return queries * reduction_matrix(mapping)



def reduction_matrix(mapping, canonical_order=False):
    """ Returns an m x n matrix R where n is the dimension of 
        the original data and m is the dimension of the reduced data.

        Reduces data vector x with R x
        Expands workload matrix W with W' R
    """
    assert mapping.ndim == 1, "Can only handle 1-dimesional mappings for now, domain should be flattened"

    unique, indices, inverse, counts = mapping_statistics(mapping)

    if canonical_order:
        mapping = canonical_ordering(mapping)

    n = mapping.size
    m = unique.size
    data = np.ones(n)
    cols = np.arange(n)
    rows = inverse

    return sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=int)


def expansion_matrix(mapping, canonical_order=False):
    """ Returns an n x m matrix E where n is the dimension of 
        the original data and m is the dimension of the reduced data.

        Expands data vector x with E x'
        Reduces workload matrix W with W E
    """
    assert mapping.ndim == 1, "Can only handle 1-dimesional mappings for now, domain should be flattened"

    unique, indices, inverse, counts = mapping_statistics(mapping)

    if canonical_order:
        mapping = canonical_ordering(mapping)

    n = mapping.size
    m = unique.size
    data = np.ones(n)
    cols = np.arange(n)
    rows = inverse

    R = sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=int)
    scale = sparse.spdiags(1.0 /counts, 0, m, m)

    return R.T * scale


def projection_matrix(mapping, idx):
    """ Returns m x n matrix P where n is the dimension of the 
        original data and m is the number of occurence of idx
        in mapping.

        :param mapping: vector with indices representing groups
        :param idx: index of group from which to create projection

        Projects vector x with P x and matrix W with W P^T
        Unprojects vector x with P^T x and matrix W with W P
    """
    mask = np.ma.masked_where(mapping!=idx, mapping).mask

    if np.all(~mask): # when all entries are False, a single False will be returned
        mask = np.array([False]*len(mapping))

    cols = np.where(~mask)[0]
    rows = np.arange(cols.size)
    vals = np.ones_like(rows)
    P = sparse.csr_matrix((vals, (rows, cols)), (rows.size, mask.size))

    return P


def combine(p1, p2):
    """ Returns p3, an (n+m) dimensional array of integers such that
        p3[i,j] = p3[i', j'] iff p1[i] = p1[i'] and p2[j] = p2[j']

        :param p1:  an n dimensional array of integers
        :param p2:  an m dimensional array of integers
    """
    _, i1 = np.unique(p1.flatten(), return_inverse=True)
    _, i2 = np.unique(p2.flatten(), return_inverse=True)
    lookup = np.arange(i1.size * i2.size).reshape(i1.size, i2.size)
    # note: cartesian product, but order is very important
    # this order works when flattening/reshaping is done in row-major form
    pairs = np.dstack(np.meshgrid(i1, i2, indexing='ij')).reshape(-1,2)
    flat = lookup[pairs[:,0], pairs[:,1]]

    return flat.reshape(p1.shape + p2.shape)


def combine_all(mappings):
    """ Returns an ndarray with each dimension corresponding to one
        of mapping.
    """
    # Note(ryan): test to make sure combine is associative
    return reduce(combine, mappings, np.ones((), dtype=int))


def extract_M(W):
    assert type(W) is sparse.csr_matrix, 'W must by csr_sparse'

    return W.getrow(W.nonzero()[0][0])


def complement(A, grid_size=None):
    '''return the queries on the complementary domain
    :param grid_size: The griding size of the new queris, if None, return total on the complementary domain
    Currently complementary domain are those indices with column norm(L1) 0.
    '''
    comp = []
    if isinstance(A, np.ndarray) is False:
        A = A.toarray()
    norm = np.linalg.norm(A,ord = 1,axis = 0)

    compl_size = len(norm) - np.count_nonzero(norm)
    grid_size = compl_size if grid_size is None else grid_size
    grid_num = int(math.ceil(compl_size/float(grid_size)))
    if grid_num==0:
        return None

    ind = 0
    for g in range(grid_num):
        q = np.zeros(len(norm))
        remain_in_group = grid_size
        while (remain_in_group>0) and (ind<len(norm)):
            if np.isclose(norm[ind],0.0):
                q[ind]=1
                remain_in_group-=1
            ind +=1
        comp.append(q)

    return sparse.csr_matrix(comp)
