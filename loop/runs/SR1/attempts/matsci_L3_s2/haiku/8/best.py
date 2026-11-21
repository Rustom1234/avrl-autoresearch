import numpy as np
def equation(eps, T, params):
    g = 1-np.exp(-params[2]*T)
    f = eps**params[5]*np.exp(-params[6]*eps*T**params[7])
    return params[0] + params[1]*g + (params[3] + params[4]*g)*f
