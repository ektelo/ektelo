import numpy as np
from ektelo import util

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
        lower_list = quick_product(*lower).T
        upper_list = quick_product(*upper).T
    except ValueError:
        # else fall back to standard crossproduct, will be slower when results size get large
        x = np.array(list(itertools.product(*boarder_list)))
        lower_list, upper_list = x.transpose([2,0,1])
    
    lower_list = lower_list + cur_range_l
    upper_list = upper_list + cur_range_l
    return lower_list, upper_list

print(np.array( split_rectangle(np.arange(16).reshape(4,4), 2,2)).transpose([1,0,2]))
print(grid_split_range( (0,0), (3,3), branching_list = [2,2]))