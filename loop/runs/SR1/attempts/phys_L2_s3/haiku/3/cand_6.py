import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*v + params[3]*x**3 + params[4]*np.cos(params[5]*t) + params[6]*np.sin(params[5]*t) + params[7]*t + params[8]*t*x + params[9]*v*np.abs(v)
