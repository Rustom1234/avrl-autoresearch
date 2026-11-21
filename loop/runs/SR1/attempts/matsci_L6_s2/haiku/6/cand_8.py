import numpy as np
def equation(eps, T, params):
    return params[0]*T**params[1]*eps**(params[2]/T + params[3])*np.exp(params[4]*T)
