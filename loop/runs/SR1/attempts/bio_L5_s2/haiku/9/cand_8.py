import numpy as np
def equation(t, P, params):
    return params[0]*np.exp(-params[1]*P) + params[2]*P + params[3] + params[4]*P**2
