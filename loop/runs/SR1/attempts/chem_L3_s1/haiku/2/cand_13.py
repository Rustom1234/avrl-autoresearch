import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 - params[3]*(np.exp(params[4]*A)-1)*(1+params[5]*t)