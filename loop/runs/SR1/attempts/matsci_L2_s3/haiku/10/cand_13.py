import numpy as np
def equation(eps, T, params):
    return params[0]*T**params[1]*np.exp(params[2]/T)*(1+params[3]*eps**2)
