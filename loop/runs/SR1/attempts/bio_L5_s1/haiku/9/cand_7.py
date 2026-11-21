import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*np.abs(P)**params[3] + params[4]*P**2
