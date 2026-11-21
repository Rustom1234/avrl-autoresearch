import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1]*t**params[2]*np.exp(-params[3]*t)