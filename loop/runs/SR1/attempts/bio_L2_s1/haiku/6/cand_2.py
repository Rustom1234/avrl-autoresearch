import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*np.exp(-params[3]*P) + params[4]*t*np.exp(-params[5]*P)
