import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*v + params[3]*v*np.abs(v) + params[4]*v**3 + params[5]*x + params[6]*x**3 + params[7]*np.cos(params[8]*t+params[9])