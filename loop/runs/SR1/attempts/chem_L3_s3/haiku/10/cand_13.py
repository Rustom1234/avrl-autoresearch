import numpy as np
def equation(t, A, params):
    return params[0]/t + params[1]*np.exp(-params[2]*t) + params[3] + params[4]*t + (params[5] + params[6]*t + params[7]*t**2)*A**2 + params[8]*A*t
