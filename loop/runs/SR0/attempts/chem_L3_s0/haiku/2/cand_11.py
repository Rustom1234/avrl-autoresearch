import numpy as np
def equation(t, A, params):
    g = params[8] + params[9]*(1-np.exp(-params[4]*t))
    return params[0]*A + params[1]*A**2 + params[2]*A**3 + params[3]*g*A + params[5]*g*A**2 + params[6]*g*A**3 + params[7]
