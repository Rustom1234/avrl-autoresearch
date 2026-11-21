import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T**params[2]*(1-np.exp(-np.abs(eps-params[3])/np.abs(params[4]))) + params[5]*T/eps + params[6]*T**2*np.sqrt(eps)
