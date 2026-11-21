import numpy as np
def equation(t, A, params):
    g = params[0]*np.exp(-params[1]*t**params[2])
    return g - params[3]*A**params[4]*t**params[5] - params[6]*A*t**params[7]*np.exp(-params[8]*t)
