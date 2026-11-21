import numpy as np
def equation(t, x, v, params):
    return -params[0]*v - params[1]*x - params[2]*x**3 + params[3]*np.cos(params[4]*t) + params[5]*np.sin(params[4]*t) + params[6]
