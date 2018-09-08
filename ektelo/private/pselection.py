from builtins import zip
import numpy as np
import math
from functools import reduce
from scipy import sparse
from ektelo import math
from ektelo.matrix import EkteloMatrix
from ektelo.operators import SelectionOperator
from ektelo.algorithm.privBayes import privBayesSelect

def noisyMax(scores, sensitivity, epsilon, prng, measuredQueries):
    """ Alternative of exponential mechanism.
        Given the same input, the probability (before normalization) that a query will be selected is 1/2
        of that of the exponential mechanism.
    """
    scores = np.array(scores, copy=True).reshape((len(scores),1))
    scale = (1 * sensitivity / epsilon)
    noise = prng.laplace(scale=scale, size=(len(scores),1))
    #print scores,measuredQueries
    # filter out previous selected queries
    for i in measuredQueries:
        scores[i] = 0.0
    noisy_scores = scores + noise
    result_index = np.argmax(noisy_scores)

    return result_index


def exponentialMechanism( scores, sensitivity, eps, prng, measuredQueries):
    """Choose the worst estimated query (set) using the exponential mechanism.

    x - true data vector
    hatx - estimated data vector
    Q - the queries to be chosen from
    epsilon - private parameter
    """

    for i in measuredQueries:
        scores[i] = 0.0
    merr = max(scores)
    # compute the sampling probability
    prob = np.exp( float(eps)* (scores - merr) / 2.0 )
    sample = prng.random_sample() * sum(prob)
    for c in range(len(prob)):
        sample -= prob[c]
        if sample <= 0:
            return c

    return len(prob)-1


class WorstApprox(SelectionOperator):
    ''' Choose privately the workload query whose estimate on xest has the greatest error.
        (Commonly used in MWEM)
    '''

    def __init__(self, W, measuredQueries, x_est, eps, mechanism="NOISYMAX"):
        super(WorstApprox, self).__init__()

        assert ((mechanism == "NOISYMAX") or (mechanism == "EXPONENTIAL")
                ), "mechanism must be set to NOISYMAX or EXPONENTIAL"

        self.W = W
        self.measuredQueries = measuredQueries
        self.x_est = x_est
        self.eps = eps
        self.mechanism = mechanism

    def select(self, x, prng):
        true_answers = self.W.dot(x)
        est_answers = self.W.dot(self.x_est)
        scores = np.abs(true_answers - est_answers)
        if (self.mechanism == "NOISYMAX"):
            index = noisyMax(scores, 1.0, self.eps, prng, self.measuredQueries)
        elif (self.mechanism == "EXPONENTIAL"):
            index = exponentialMechanism(scores, 1.0, self.eps, prng, self.measuredQueries)

        ans = self.W[index]
        ans.mwem_index = index # Note(ryan): a bit of a hack, try to fix in the future 
        return ans


class PrivBayesSelect(SelectionOperator):

    def __init__(self, theta, domain_shape, eps):
        super(PrivBayesSelect, self).__init__()

        self.theta = theta
        self.domain_shape = domain_shape
        self.eps = eps

    @staticmethod
    def make_models(model_str):
        X  = []
        for line in model_str.split('\n'):
            if line == '':
                continue
            numbers = line.strip().split(',')
            x = []
            for i in range(len(numbers) // 2):
                attribute = int(numbers[i*2])
                dsize = int(numbers[i*2 + 1])
                pair = (attribute, dsize)
                x.append(pair)
            X.append(x)

        return X

    '''
    Builds an n x m matrix that can be used to transform histograms of size n to histograms of size m
    It preserves scale: all columns sum up to 1
    It preserves shape: all rows sum up to m / n

    Note that in order to preserve shape a uniformity assumption has been made
    on each bucket of the histograms.

    Also note that the specific way it assigns weights to the elements of P
    is not a heuristic but an exact way to preserve uniformity.
    '''
    @staticmethod
    def domain_transform(n, m):
        P = np.zeros((n, m))
        ratio = np.divide(m,n, dtype = float)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                colsum = P.sum(axis = 0)[j]
                rowsum = P.sum(axis = 1)[i]
                # If colsum == 1 then skip. 
                #I added the condition >= .99 since when the numbers are not powers of 2 arithmetics is never exact (e.g., 1/3 * 3 != 1)
                if (colsum >= 0.99) or (rowsum >= ratio - 0.01):
                    continue
                else:
                    P[i][j] = np.min([1 - colsum, ratio - rowsum, 1])
        return sparse.csr_matrix(P)

    @staticmethod
    def get_measurements(model, domain_shape):
        # model is a set of contingency tables to calculate
        # each contingency table is a list of [(attribute, size)] 
        M = []
        for table in model:
            Q = [np.ones((1,size)) for size in domain_shape]
            for attribute, size in table:
                full_size = domain_shape[attribute]
                I = sparse.identity(size) 
                if size != full_size:
                    P = PrivBayesSelect.domain_transform(size, full_size)
                    Q[attribute] = I * P
                elif size == full_size:
                    Q[attribute] = I
                else:
                    print('bug here')
            M.append(reduce(sparse.kron, Q))
        return math.vstack(M)

    @staticmethod
    def get_config_str(relation):
        config = relation.config
        config_str = ''

        for column in config:
            d_left, d_right = config[column]['domain']
            if config[column]['type'] == 'continuous':
                config_str +='C '
                config_str += str(d_left) 
                config_str += ' '
                config_str += str(d_right)
                config_str += ' '
            elif config[column]['type'] == 'discrete':
                config_str += 'D '
                for i in range(d_left, d_right):
                    config_str += str(i)
                    config_str += ' '
            config_str += '\n'

        return config_str

    def select(self, x, prng):
        relation = x

        seed = prng.randint(1E4, 1E9)
        # convert config to privBayes format
        config_str = self.get_config_str(relation) 

        model_str = privBayesSelect.py_get_model(
            np.ascontiguousarray(relation.df.astype(np.int32)), 
            config_str.encode('utf-8'), 
            self.eps, 
            self.theta,
            seed)
        
        model = PrivBayesSelect.make_models(model_str.decode('utf-8'))
        M = PrivBayesSelect.get_measurements(model, self.domain_shape)

        return EkteloMatrix(M)
