import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1]*t**params[2]*np.exp(-params[3]*t) + params[4]*A**params[5] + params[6]*A**params[7]*np.exp(-params[8]*t)
