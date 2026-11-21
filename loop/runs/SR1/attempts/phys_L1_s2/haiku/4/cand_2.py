import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*x + params[3]*v + params[4]*x**3 + params[5]*x**2 + params[6]*v*np.abs(v) + params[7]*np.sin(t) + params[8]*np.cos(t) + params[9]*x*v