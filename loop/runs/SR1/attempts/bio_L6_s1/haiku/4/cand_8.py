import numpy as np
def equation(t, P, params):
    K = params[1] + params[2]*t
    return params[0]*np.sign(K-P)*np.abs(1 - P/K)**params[3]*(1 + params[4]*t)