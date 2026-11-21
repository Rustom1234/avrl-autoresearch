import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*eps*np.exp(-params[3]*eps) + params[4]*eps**2 + params[5]*T*eps + params[6]*eps**3
