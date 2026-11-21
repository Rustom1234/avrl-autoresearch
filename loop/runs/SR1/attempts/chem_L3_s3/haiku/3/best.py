import numpy as np
def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t) - params[2]*(A*t)**params[3] + params[4]*np.exp(-params[5]*t)*A**params[6]