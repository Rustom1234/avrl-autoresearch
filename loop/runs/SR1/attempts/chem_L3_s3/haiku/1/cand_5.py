import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*t*A**params[3] + params[4]*A**params[5] + params[6]*t + params[7]