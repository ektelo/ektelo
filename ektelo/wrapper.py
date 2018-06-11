from __future__ import absolute_import
from .client import inference
from .client import selection
from .client import mapper
from .client import measurement as cmeasurement
from .private import measurement
from .private import pselection


def identity(shape):
    return selection.Identity(shape).select()


def h2(shape):
    return selection.H2(shape).select()


def hb(shape):
    return selection.HB(shape).select()


def greedyH(shape, W):
    return selection.GreedyH(shape, W).select()


def total(shape):
    return selection.Total(shape).select()


def quad_tree(shape):
    return selection.QuadTree(shape).select()


def agrid_select(shape, x, eps, c2=5):
    return selection.AdaptiveGrid(shape, x, eps, c2).select()


def ugrid_select(shape, x_sum, eps, ag_flag=False, c=10, gz=0):
    return selection.UniformGrid(shape, x_sum, eps, ag_flag, c, gz).select()


def least_squares(Ms, ys, scale_factors=None):
    return inference.LeastSquares().infer(Ms, ys, scale_factors)


def non_negative_least_squares(Ms, ys, scale_factors=None):
    return inference.NonNegativeLeastSquares().infer(Ms, ys, scale_factors)


def multiplicative_weights(Ms, ys, x_est, update_rounds=50, scale_factors=None):
    return inference.MultiplicativeWeights(update_rounds).infer(Ms, ys, x_est, scale_factors)


def laplace_scale_factor(M, eps):
    return cmeasurement.laplace_scale_factor(M, eps)


def non_negative_least_squares(Ms, ys, scale_factors=None):
    return inference.NonNegativeLeastSquares().infer(Ms, ys, scale_factors)


def ugrid_mapper(shape, x_sum, eps, ag_flag=False, c=10, gz=0):
    return mapper.UGridPartition(shape, x_sum, eps, ag_flag, c, gz).mapping()


def striped(domain, stripe_dim):
    return mapper.Striped(domain, stripe_dim).mapping()
