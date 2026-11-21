import numpy as np
def equation(eps, T, params):
    s0 = params[0] + params[1]*T
    ss = params[2] + params[3]*T**2
    k = params[4] + params[5]*T
    return -(ss - (ss - s0)*np.exp(-k*eps))
