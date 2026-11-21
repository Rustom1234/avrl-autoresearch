import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*A**3 + params[2]*t**params[3]*np.exp(-params[4]*t)
