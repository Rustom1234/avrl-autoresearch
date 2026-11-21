import numpy as np
def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t) + params[2]*np.sin(params[1]*t) + params[3] - params[4]*v - params[5]*np.sin(x) - params[6]*np.cos(x) - params[7]*x**3 - params[8]*x - params[9]*t
