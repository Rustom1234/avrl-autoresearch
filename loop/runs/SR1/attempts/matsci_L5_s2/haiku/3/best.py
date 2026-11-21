import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*(1-np.exp(-params[2]*eps)) + params[3]*eps**params[4] + params[5]*T*eps**params[6]
