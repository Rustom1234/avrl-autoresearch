import numpy as np
def equation(t, A, params):
    k = params[0]*np.exp(-params[1]*t)
    return -k*(1 - (A/params[2])**params[3])
