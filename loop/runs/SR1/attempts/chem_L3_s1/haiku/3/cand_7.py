import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*A**params[2] - params[3]*A**params[4] + params[5]*np.exp(-params[6]*t)
