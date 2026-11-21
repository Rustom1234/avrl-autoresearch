import numpy as np
def equation(eps, T, params):
    p = params[9]
    return (params[0] + params[1]*T + params[2]*T**2) + (params[3] + params[4]*T + params[5]*T**2)*np.abs(eps)**p
