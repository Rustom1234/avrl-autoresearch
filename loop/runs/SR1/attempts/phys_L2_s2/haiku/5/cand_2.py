import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*v*np.abs(v) + params[5]*np.cos(params[6]*t) + params[7]*np.sin(params[6]*t) + params[8]*x*v
