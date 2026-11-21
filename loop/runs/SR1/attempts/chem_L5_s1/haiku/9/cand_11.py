import numpy as np
def equation(t, A, params):
    p = 1.0/(1.0+params[1]*t)
    return -params[0]*(A*np.exp(-A))**p
