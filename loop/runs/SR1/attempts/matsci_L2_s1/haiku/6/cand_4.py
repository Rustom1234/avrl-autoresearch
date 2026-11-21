import numpy as np
def equation(eps, T, params):
    A = params[0]*(1-np.exp(-params[1]*T))
    B = params[2]*(1-np.exp(-params[3]*T))
    return -(A + params[4]) + B*eps + params[5]*eps**2