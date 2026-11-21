import numpy as np
def equation(t, A, params):
    return -params[0]*np.exp(params[1]*t)*A + params[2]*np.exp(params[3]*t)*A**2
