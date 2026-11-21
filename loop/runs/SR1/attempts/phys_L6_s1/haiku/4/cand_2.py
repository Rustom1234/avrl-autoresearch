import numpy as np
def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t) + params[2]*np.sin(params[1]*t) - params[3]*v - params[4]*x - params[5]*x**3 - params[6]*v**3 - params[7]*x**2 + params[8]*np.sin(params[9]*t)