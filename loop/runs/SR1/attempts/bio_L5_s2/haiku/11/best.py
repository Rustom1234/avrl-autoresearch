import numpy as np
def equation(t, P, params):
    return params[0]*np.exp(-params[1]*P) + params[2]*P**3 + params[3]*P + params[4]*t*P**3 + params[5]*t*np.exp(-params[1]*P)