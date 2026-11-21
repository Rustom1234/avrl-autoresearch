import numpy as np
def equation(eps, T, params):
    g = 1-np.exp(-params[0]*T)
    return params[1] + g*(params[2] + params[3]*eps**params[4]*np.exp(-params[5]*eps))
