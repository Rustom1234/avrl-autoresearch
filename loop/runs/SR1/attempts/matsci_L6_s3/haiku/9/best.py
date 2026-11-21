import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps + params[2]*np.log(T) + params[3]*np.exp(-eps/T) + params[4]*eps*np.exp(-T)
