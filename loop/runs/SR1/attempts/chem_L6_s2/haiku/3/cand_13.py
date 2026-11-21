import numpy as np
def equation(t, A, params):
    return -params[0]*A**params[1] + params[2]*t**params[3]*np.exp(-params[4]*A) + params[5]*A*np.exp(-params[6]*t) + params[7]