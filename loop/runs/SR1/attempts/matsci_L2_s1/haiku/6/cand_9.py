import numpy as np
def equation(eps, T, params):
    A = params[0] + params[1]*np.exp(-params[2]*T)
    B = params[3] + params[4]*np.exp(-params[5]*T)
    return -A + B*eps**params[6]