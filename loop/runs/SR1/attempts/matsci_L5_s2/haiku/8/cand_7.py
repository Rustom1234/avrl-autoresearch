import numpy as np
def equation(eps, T, params):
    return params[0]*eps**params[1]*np.exp(-params[2]*eps)+params[3]*eps