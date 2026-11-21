import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*np.exp(-params[3]*t)*A + params[4]*A + params[5]*A**2 + params[6]*t*A + params[7]*A**2*t
