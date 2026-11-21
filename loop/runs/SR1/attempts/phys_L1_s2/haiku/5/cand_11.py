import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t**params[2] + params[3]*x + params[4]*x**params[5] + params[6]*v + params[7]*x**3 + params[8]*v*np.abs(v)
