import numpy as np
def equation(eps, T, params):
    return params[0]*T**params[1] + params[2]*T**params[3]*(1-np.exp(-params[4]*eps**params[5]))
