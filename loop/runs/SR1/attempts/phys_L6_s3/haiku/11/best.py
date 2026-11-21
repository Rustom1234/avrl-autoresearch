import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*v*np.abs(v) + params[5]*v**3 + params[6]*np.cos(params[8]*t) + params[7]*np.sin(params[8]*t) + params[9]*v*t
