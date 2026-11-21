import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1]*(1-np.exp(-params[2]*t))*np.exp(-params[3]*A*t) - params[4]*A**params[5]
