import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*np.exp(-params[3]*t)/A**params[4] + params[5]*A**params[6]*np.exp(-params[7]*t) + params[8]*A + params[9]
