import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*np.sqrt(t) + params[2]*t + params[3]*A**params[4] + params[5]*np.sqrt(t)*A**params[6]
