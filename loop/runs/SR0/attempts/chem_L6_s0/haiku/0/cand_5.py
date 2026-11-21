import numpy as np
def equation(t, A, params):
    return params[0]*t**params[1] + params[2]*A**params[3]*t**params[4] + params[5]*A**params[6]*np.exp(-params[7]*t)