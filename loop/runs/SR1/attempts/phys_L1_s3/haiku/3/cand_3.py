import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]/x + params[4]*np.sin(params[5]*t+params[6]) + params[7]*v
