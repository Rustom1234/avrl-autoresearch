import numpy as np
def equation(eps, T, params):
    A = params[0] + params[1]*np.exp(-params[2]*T)
    return -A + (params[3] + params[4]*T + params[5]*T**2)*eps + (params[6] + params[7]*T)*eps**2