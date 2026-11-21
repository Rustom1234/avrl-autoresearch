import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*np.log(t) + params[3]/t + params[4]*A*np.exp(-params[5]*t) + params[6]*np.exp(-params[7]*t)
