import numpy as np
def equation(t, A, params):
    p = params[2]/(1.0+params[1]*t)
    return -params[0]*(A*np.exp(-params[3]*A))**p
