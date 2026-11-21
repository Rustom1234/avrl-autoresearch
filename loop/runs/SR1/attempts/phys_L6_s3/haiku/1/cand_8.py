import numpy as np
def equation(t, x, v, params):
    return -params[0]*v - params[1]*x - params[2]*x**3 + params[3]/v + params[4]*np.cos(params[5]*t) + params[6]*np.sin(params[5]*t) + params[7]
