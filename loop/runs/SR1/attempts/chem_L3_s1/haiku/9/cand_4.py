import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*(1 - A/params[2]) + params[3]*np.exp(-params[4]*t)*A**2 
