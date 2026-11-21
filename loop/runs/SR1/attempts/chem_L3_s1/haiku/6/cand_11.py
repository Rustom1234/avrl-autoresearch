import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*np.exp(params[3]*A) + params[4]*t*A**2 + params[5]*A**2
