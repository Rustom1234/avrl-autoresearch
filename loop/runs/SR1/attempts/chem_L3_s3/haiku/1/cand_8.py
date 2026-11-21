import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*A*(1-np.exp(-params[3]*t)) + params[4]*A**2 + params[5]*A**2*np.exp(-params[3]*t) + params[6]