import numpy as np
def equation(eps, T, params):
    return params[0]*eps + params[1]*T**params[2] + params[3]*eps*T + params[4]*(1-np.exp(-params[5]*eps)) + params[6]*T*(1-np.exp(-params[5]*eps))
