import numpy as np
def equation(t, A, params):
    return params[0]*A + params[1]*A**2 + params[2]*t*np.exp(-params[3]*t) + params[4] + params[5]*t
