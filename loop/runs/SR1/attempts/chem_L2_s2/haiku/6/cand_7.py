import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*A**params[2] + params[3]*A*np.exp(-params[4]*t) + params[5]*A**2*np.exp(-params[6]*t)