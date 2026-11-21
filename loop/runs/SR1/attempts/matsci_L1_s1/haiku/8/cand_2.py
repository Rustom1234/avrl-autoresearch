import numpy as np
def equation(eps, T, params):
    return params[0]*T**params[1]*np.cos(params[2]*eps*T**params[3]) + params[4]*T**params[5]
