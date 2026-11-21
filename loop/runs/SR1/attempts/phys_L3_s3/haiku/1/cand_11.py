import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*x + params[4]*x**3 + params[5]*v + params[6]*v**3 + params[7]*v*np.abs(v) + params[8]*x*v**2
