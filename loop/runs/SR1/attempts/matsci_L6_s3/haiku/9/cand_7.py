import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.exp(-params[2]*T) + (params[3] + params[4]*T**2)*eps*np.exp(-params[5]*eps)
