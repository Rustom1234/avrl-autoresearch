import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*x + params[3]*v + params[4]*x**3 + params[5]*v*np.abs(v) + params[6]*np.sin(params[7]*x) + params[8]*t*v + params[9]*np.sqrt(t)