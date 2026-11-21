import numpy as np
def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*np.exp(-params[2]*t) + params[3] - params[4]*A**params[5]*t**params[6]
