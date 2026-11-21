import numpy as np
def equation(t, A, params):
    k = params[0]*np.exp(params[1]*t)
    return -k*np.abs(A)**params[2]*(1 - A/(params[3]+params[4]*t))