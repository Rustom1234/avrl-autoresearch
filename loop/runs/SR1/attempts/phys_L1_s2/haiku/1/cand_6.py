import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*np.cos(params[3]*t+params[4]) + params[5]*x + params[6]*x**2 + params[7]*x**3 + params[8]*v + params[9]*v*np.abs(v)
