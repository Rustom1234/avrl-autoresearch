import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T**2 + params[2]*T) * (1 - np.exp(-params[3]*eps)) + params[4]*eps**params[5]*np.exp(-params[6]*T) + params[7]*T
