import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t*t + params[3]*np.exp(params[4]*t) + params[5]/x
