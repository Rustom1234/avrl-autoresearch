import numpy as np
def equation(t, A, params):
    return -params[0]*A**params[1]*np.exp(-params[2]*t) + params[3]*np.exp(-params[4]*t) + params[5]*A*np.exp(-params[6]*t) + params[7]*A**2*np.exp(-params[8]*t)
