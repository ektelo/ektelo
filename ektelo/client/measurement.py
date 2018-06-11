from ektelo import util
from ektelo.private.measurement import Laplace
import math


def laplace_scale_factor(A, eps):
    sensitivity = Laplace.sensitivity_L1(A)
    laplace_scale = util.old_div(sensitivity, float(eps))

    return math.sqrt(2.0*laplace_scale**2)
