import numpy as np
def equation(t, A, params):
    S = params[0]*t**params[1]*np.exp(-params[2]*t**params[3])
    return S - params[4]*A**params[5]*t**params[6]
