import numpy as np
def equation(t, A, params):
    return params[0]*t*np.exp(-params[1]*t)*np.sqrt(A)
