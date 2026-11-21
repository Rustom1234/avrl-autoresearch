import numpy as np
def equation(eps, T, params):
    s0 = params[0] + params[1]*T
    k = params[2] + params[3]*T
    m = params[4] + params[5]*T
    return s0 + k*eps**params[6]*np.exp(-m*eps)
