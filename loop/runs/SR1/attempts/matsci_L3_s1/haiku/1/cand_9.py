import numpy as np
def equation(eps, T, params):
    A = params[0] + params[1]*T
    B = params[2] + params[3]*T
    m = params[4] + params[5]*T
    n = params[6]
    return A*np.exp(-m*eps) + B*eps**n*np.exp(-m*eps) + params[7]*eps
