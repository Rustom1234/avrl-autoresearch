import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.sqrt(T) + params[2]*eps**1.5