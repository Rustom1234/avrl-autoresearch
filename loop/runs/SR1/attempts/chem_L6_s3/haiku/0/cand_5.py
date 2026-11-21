import numpy as np
def equation(t, A, params):
    S = params[0]*(1-np.exp(-params[1]*t))*np.exp(-params[2]*t)
    return S - params[3]*A**params[4]*t**params[5]
