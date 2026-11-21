import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps**2 + params[2]*T + params[3]*T**2 + params[4]*T**3 + params[5]*np.exp(-T)