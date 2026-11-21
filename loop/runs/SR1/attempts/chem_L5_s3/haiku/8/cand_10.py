import numpy as np
def equation(t, A, params):
    return params[0]*A + params[1]*A**2 + params[2]*(1-np.exp(-params[3]*t))*np.exp(-params[4]*t) + params[5] + params[6]*t
