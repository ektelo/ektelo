from __future__ import division
import numpy as np
import math
from ektelo import matrix
from scipy import linalg, optimize
from scipy.sparse.linalg import lsmr, lsqr
from scipy import sparse
import ektelo
from ektelo import util
from ektelo.operators import InferenceOperator


def get_A(M, noise_scales):
    """
        Calculate matrix 'A' of measurements, scaled appropriately for inference

    """
    sf = (util.old_div(1.0, np.array(noise_scales)))    # reciprocal of each noise scale
    D = ektelo.math.diag_like(M, sf, 0, sf.size, sf.size)
    return D * M  # scale rows


def get_y(ans, noise_scales):
    """
        Calculate 'y' of answers, scaled appropriately for inference
    """
    sf = (util.old_div(1.0, np.array(noise_scales)))    # reciprocal of each noise scale
    y = ans * sf                      # element-wise multiplication
    y = y[:, np.newaxis]          # make column vector
    return y


def nls_lbfgs_b(A, y, l1_reg=0.0, l2_reg=0.0, maxiter = 15000):
    """
    Solves the NNLS problem min || Ax - y || s.t. x >= 0 using gradient-based optimization
    :param A: numpy matrix, scipy sparse matrix, or scipy Linear Operator
    :param y: numpy vector
    """
    M = sparse.linalg.aslinearoperator(A)

    def loss_and_grad(x):
        diff = M.matvec(x) - y
        res = 0.5 * np.sum(diff ** 2)
        f = res + l1_reg*np.sum(x) + l2_reg*np.sum(x**2)
        grad = M.rmatvec(diff) + l1_reg + l2_reg*x

        return f, grad

    xinit = np.zeros(A.shape[1])
    bnds = [(0,None)]*A.shape[1]
    xest,_,info = optimize.lbfgsb.fmin_l_bfgs_b(loss_and_grad,
                                                x0=xinit,
                                                pgtol=1e-4,
                                                bounds=bnds,
                                                maxiter=maxiter,
                                                m=1)
    xest[xest < 0] = 0.0
    return xest, info


def nls_slsqp(A, y, known_total):
    N = A.shape[1]
    M = sparse.linalg.aslinearoperator(A)

    def loss_and_grad(x):
        diff = M.matvec(x) - y
        res = 0.5 * np.sum(diff ** 2)

        return res, M.rmatvec(diff)

    xinit = np.zeros(N)
    bnds = [(0,None)]*N
    cons = {'type':'eq',
            'fun':lambda x: x.sum()-known_total,
            'jac':lambda _: np.ones(N)}
    opts = { 'maxiter' : 10000 }
    res = optimize.minimize(loss_and_grad,
                            xinit,
                            method='SLSQP',
                            jac=True,
                            bounds=bnds,
                            constraints=cons,
                            options=opts)

    return res.x


def eval_x(hatx, q):
    """evaluation of a query in the form of t"""
    return float(q.dot(hatx))


def multWeightsUpdate(hatx, Q, Q_est, updateRounds = 1):
    """ Multiplicative weights update, supporting multiple measurements and repeated update rounds
    hatx: starting estimate of database
    Q: list of query arrays representing measurements
    Q_est: list of corresponding answers to query
    updateRounds: number of times to repeat the update of _all_ provided queries
    """
    assert Q.shape[0]==len(Q_est)

    total = sum(hatx)

    if not isinstance(Q, sparse.csc_matrix):
        Q = sparse.csr_matrix(Q)

    for i in range(updateRounds):
        for q_est, q in zip(Q_est, Q):
            error = q_est - eval_x(hatx,q)        # difference between query ans on current estimated data and the observed answer
            update_vector = np.exp( q.toarray() * error / (2.0 * total) ).flatten()  # note that numpy broadcasting is happening here
            hatx = hatx * update_vector
            hatx = hatx * total / sum(hatx)     # normalize

    return hatx


class ScalableInferenceOperator(InferenceOperator):

    def _apply_scales(self, Ms, ys, scale_factors):
        if scale_factors is None:
            M = Ms
            y = ys
            noise_scales = [1.0] * len(y)
        else:
            assert type(Ms) == list and type(ys) == list
            assert len(Ms) > 0 and len(Ms) == len(ys) and len(ys) == len(scale_factors)
        
            M = None
            y = []
            noise_scales = []

            for i in range(len(scale_factors)):
                if M is None:
                    M = Ms[i]
                else:
                    M = matrix.VStack((M, Ms[i]))
                y = np.concatenate((y, ys[i]))
                noise_scales = np.concatenate((noise_scales, [scale_factors[i]] * len(ys[i])))

        return get_A(M, noise_scales), get_y(y, noise_scales).flatten()


class LeastSquares(ScalableInferenceOperator):

    def __init__(self, method='lsmr', l2_reg=0.0, known_total=None, stein=False):
        super(LeastSquares, self).__init__()

        self.method = method
        self.l2_reg = l2_reg
        self.known_total = known_total
        self.stein = stein

    def __known_total_problem(self, A, y):
        m, n = A.shape
        A0, A1 = A[:, :n - 1], A[:, n - 1]
        if type(A) == np.ndarray:
            z = y - A1 * self.known_total
            return A0 - A1.reshape(m, 1), z
        else:
            A1 = A1.toarray().flatten()
            z = y - A1 * self.known_total

        def matvec(v):
            return A0.dot(v) - A1 * v.sum()

        def rmatvec(v):
            return A0.T.dot(v) - A1.dot(v)
        B = LinearOperator(shape=(m, n - 1), matvec=matvec,
                           rmatvec=rmatvec, dtype=np.float64)
        return B, z

    def infer(self, Ms, ys, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        A, y = self._apply_scales(Ms, ys, scale_factors)

        if self.known_total is not None:
            A, y = self.__known_total_problem(A, y)

        if self.method == 'standard':
            assert self.l2_reg == 0, 'l2 reg not supported with method=standard'
            assert isinstance(
                A, np.ndarray), "method 'standard' only works with dense matrices"
            (x_est, _, rank, _) = linalg.lstsq(A, y, lapack_driver='gelsy')
        elif self.method == 'lsmr':
            res = lsmr(A, y, atol=0, btol=0, damp=self.l2_reg)
            x_est = res[0]
        elif self.method == 'lsqr':
            res = lsqr(A, y, atol=0, btol=0, damp=self.l2_reg)
            x_est = res[0]

        if self.known_total is not None:
            x_est = np.append(x_est, self.known_total - x_est.sum())

        x_est = x_est.reshape(A.shape[1])  # reshape to match shape of x

        # James-Stein estimation
        if self.stein and x_est.size >= 3:
            adjustment = 1.0 - util.old_div((x_est.size - 2), (x_est**2).sum())
            x_est *= adjustment

        return x_est


class NonNegativeLeastSquares(ScalableInferenceOperator):
    '''
    Non negative least squares (nnls)
    Note: undefined behavior when system is under-constrained
    '''

    def __init__(self, method='LB', lasso=None):
        '''
        :param method: method for solving nnls
            'AS' for Active Set method (only for dense matrices)
            'LB' for L-BFGS-B algorithm (default; good for sparse and dense)
            'TRF' for Trust Region Reflective algorithm (best with sparse matrices)
        :param useAll: flag to use all measurements or most recent measurements
        :param lasso:
            None for no regularization
            True for regularization as determined by total estimate give by least squares
            positive number for regularization strength (xest will have sum approximately equal to this number)
        '''
        super(NonNegativeLeastSquares, self).__init__()

        self.method = method
        self.lasso = lasso

    def infer(self, Ms, ys, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        A, y = self._apply_scales(Ms, ys, scale_factors)

        if self.method == 'AS':
            assert isinstance(
                A, numpy.ndarray), "method 'AS' only works with dense matrices"
            x_est, _ = optimize.nnls(A, y)
        elif self.method == 'LB':
            if self.lasso is None:
                x_est, info = nls_lbfgs_b(A, y)
            if self.lasso:
                lasso = max(0, lsmr(A, y)[0].sum()
                            ) if self.lasso is True else self.lasso
                x_est = nls_slsqp(A, y, lasso)
        elif self.method == 'TRF':
            x_est = optimize.lsq_linear(
                A, y, bounds=(0, numpy.inf), tol=1e-3)['x']
        x_est = x_est.reshape(A.shape[1])  # reshape to match shape of x

        return x_est


class MultiplicativeWeights(InferenceOperator):
    '''
    Multiplicative weights update with multiple update rounds and optional history
    useHistory is no longer available inside the operator. To use history measurements,
    use M and ans with full history.
    '''

    def __init__(self, updateRounds=50):
        super(MultiplicativeWeights, self).__init__()
        self.updateRounds = updateRounds

    def __consolidate(self, Ms, ys, scale_factors):
        if scale_factors is None:
            M = Ms
            y = ys
            noise_scales = [1.0] * len(y)
        else:
            assert type(M) == list and type(ys) == list
            assert len(Ms) == len(ys) and len(ys) == len(scale_factors)
        
            M = None
            y = []
            noise_scales = []

            for i in range(len(scale_factors)):
                if M is None:
                    M = M_i
                else:
                    M = matrix.VStack((M, Ms[i]))
                y = np.concatenate((y, ys[i]))
                noise_scales = np.concatenate((noise_scales, [scale_factors[i]] * len(y_i)))

        return M, y, noise_scales

    def infer(self, Ms, ys, x_est, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        M, y, noise_scales = self.__consolidate(Ms, ys, scale_factors)

        """ mult_weights is an update method which works on the original domain"""
        assert x_est is not None, 'Multiplicative Weights update needs a starting xest, but there is none.'

        diff = np.array(noise_scales) - max(noise_scales)
        assert np.allclose(diff, np.zeros_like(
            diff)), 'Warning: Measurements have different noise scales but MW cannot handle this properly'

        x_est = multWeightsUpdate(x_est, M, y, self.updateRounds)
        return x_est


class AHPThresholding(ScalableInferenceOperator):
    '''
    Special update operator for AHP thresholding step.
    This operator assumes that the previous one is a Laplace measurement of the Identity workload.
    The xest is updated by answers from the Identity workload after thresholding. 
    To calculate the threshold, the eps used for the measurement is assumed to be ratio*_eps_total
    '''

    def __init__(self, eta, ratio):
        super(AHPThresholding, self).__init__()
        self.eta = eta
        self.ratio = ratio

    def infer(self, Ms, ys, eps_par, scale_factors=None):
        ''' Either:
            1) Ms is a single M and ys is a single y 
               (scale_factors ignored) or
            2) Ms and ys are lists of M matrices and y vectors
               and scale_factors is a list of the same length.
        '''
        A, y = self._apply_scales(Ms, ys, scale_factors)

        eps = eps_par * self.ratio
        x_est = lsmr(A, y.flatten())[0]
        x_est = x_est.reshape(A.shape[1])
        n = len(x_est)
        cutoff = self.eta * math.log(n) / eps
        x_est = np.where(x_est <= cutoff, 0, x_est)

        return x_est
