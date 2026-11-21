import numpy as np
def equation(eps, T, params):
    L = np.log(T)
    return params[0] + params[1]*eps**params[2] + (params[3] + params[4]*eps**params[5])*L + params[6]*L**2