import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*np.sqrt(t) + params[3]*np.log(t) + params[4]*np.exp(-params[5]*t) + params[6]*A*np.exp(-params[7]*t) + params[8]*A
