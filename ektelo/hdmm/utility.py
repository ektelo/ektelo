import math
import numpy as np

def discretize(strategy, digits=2):
    """
    Discretize a strategy matrix for use in geometric mechanism
    
    :param strategy: the strategy to discretize.  May be a 2D numpy array for an explicit strategy 
    or a list of 2D numpy arrays for a kronecker product strategies 
    :param digits: the number of digits to truncate to
    """
    if type(strategy) is np.ndarray:
        return np.round(strategy*10**digits).astype(int)
    elif type(strategy) is list:
        return [discretize(S, digits) for S in strategy]

def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)
    
