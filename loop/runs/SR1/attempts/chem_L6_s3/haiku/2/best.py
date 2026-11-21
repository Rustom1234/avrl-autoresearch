import numpy as np
def equation(t, A, params):
    return params[0]*t**params[1]*np.exp(-params[2]*t) - (params[3]+params[4]*t)*A**params[5]
