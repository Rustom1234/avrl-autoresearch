import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]/t + params[4]*np.exp(-t) + params[5]*np.exp(-t)/t + (params[6] + params[7]*t)*A**2*t**params[8]