import numpy as np
def equation(t, A, params):
    return params[0]*A + params[1]*A**2 + params[2] + params[3]*t**2 + params[4]*t**params[5]/A**params[6] + params[7]*t**2/A + params[8]*np.exp(params[9]*t)/A
