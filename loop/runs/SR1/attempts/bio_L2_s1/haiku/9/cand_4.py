import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t - params[2]*np.abs(P)**params[3] + params[4]*t*P + params[5]*t*np.abs(P)**params[6]
