import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x**3 + params[5]*v*np.abs(v) + params[6]*t + params[7]*t*x + params[8]*t*v + params[9]*t*x**2