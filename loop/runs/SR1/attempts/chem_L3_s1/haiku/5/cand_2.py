import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*A + params[3]*A**2 + params[4]*A*t + params[5]*np.exp(-params[1]*t)*A
