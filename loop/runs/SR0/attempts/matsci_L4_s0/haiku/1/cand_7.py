import numpy as np
def equation(eps, T, params):
    x = np.abs(eps - params[2])
    return params[0]*T**params[1]*x**params[3]*np.exp(-params[4]*eps) + params[5]
