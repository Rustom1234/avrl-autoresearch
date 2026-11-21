import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.sqrt(T) + params[2]*T + params[3]*T**2 + params[4]*eps**2*np.sqrt(T) + params[5]*T**3
