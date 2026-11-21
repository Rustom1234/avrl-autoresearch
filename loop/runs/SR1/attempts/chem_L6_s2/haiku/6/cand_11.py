import numpy as np
def equation(t, A, params):
    a = params[0]*np.exp(-params[1]*t)
    b = params[2]*np.exp(-params[3]*t)
    c = params[4]*np.exp(-params[5]*t)
    d = params[6]*np.exp(-params[7]*t)
    return -a/A + b - c*A + d*A**2
