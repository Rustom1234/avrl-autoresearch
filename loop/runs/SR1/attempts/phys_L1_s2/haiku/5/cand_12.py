import numpy as np
def equation(t, x, v, params):
    return params[0]*np.sqrt(t) + params[1]*np.sqrt(x) + params[2]*v*np.abs(v)
