import numpy as np
def equation(eps, T, params):
    S = params[0]*(1 - params[1]*np.exp(-params[2]*T))
    return -S + params[3]*eps**params[4]*np.exp(-params[5]*T)
