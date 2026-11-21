import numpy as np
def equation(eps, T, params):
    s0 = params[0]*np.exp(params[1]*T)+params[2]
    K = params[3]+params[4]*T
    return -(s0 + K*eps**params[5])