import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) - params[2]*np.exp(-params[3]*t)*A**2 + params[4] + params[5]*A**2 + params[6]*t
