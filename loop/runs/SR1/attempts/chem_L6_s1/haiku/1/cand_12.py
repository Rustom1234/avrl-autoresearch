import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A**2 + params[3]*np.exp(params[4]*t)*(1.0/A + params[5]) + params[6]*t**2 + params[7]*t
