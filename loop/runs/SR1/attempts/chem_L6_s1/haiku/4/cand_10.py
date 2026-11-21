import numpy as np
def equation(t, A, params):
    return params[0]*A + params[1]*A**2 + params[2] + params[3]*t**2 + (params[4] + params[5]*np.exp(params[6]*t))/A
