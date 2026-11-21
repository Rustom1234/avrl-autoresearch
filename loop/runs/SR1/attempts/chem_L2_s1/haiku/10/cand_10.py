import numpy as np
def equation(t, A, params):
    k = params[0]*np.exp(params[1]*t)
    K = params[2]*np.exp(params[3]*t)
    return -k*A/(1+K*A)
