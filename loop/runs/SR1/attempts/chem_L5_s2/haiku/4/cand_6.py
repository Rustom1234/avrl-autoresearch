import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*np.log(t) + params[2]*A + params[3]*A*np.log(t)
