import numpy as np
def equation(t, x, v, params):
    return params[0]*np.exp(-params[1]*t) + params[2]*v + params[3]*x + params[4]*x**2 + params[5]*v*np.abs(v) + params[6]*x*v + params[7]*x**3 + params[8]*v**2 + params[9]