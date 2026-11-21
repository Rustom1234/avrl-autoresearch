import numpy as np
def equation(t, A, params):
    k = params[0]*np.exp(params[1]*t)
    return -k*A/(1+params[2]*A)
