import numpy as np
def equation(eps, T, params):
    return params[0]*np.exp(-params[1]*T) - params[2]*np.sqrt(eps) + params[3]*np.exp(-params[4]*T)*np.sqrt(eps)
