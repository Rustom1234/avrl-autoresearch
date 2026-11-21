import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*np.exp(-params[3]*t) + params[4] - (params[5]*(1-np.exp(-params[6]*t)) + params[7]*t)*A**2
