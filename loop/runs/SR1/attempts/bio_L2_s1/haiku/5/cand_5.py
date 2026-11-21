import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t - (params[2] + params[3]*t)*np.abs(P)**params[4]
