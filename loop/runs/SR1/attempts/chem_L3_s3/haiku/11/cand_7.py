import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*np.exp(-params[2]*t) + params[3]*t + (params[4] + params[5]*t)*A**2 + params[6]*A
