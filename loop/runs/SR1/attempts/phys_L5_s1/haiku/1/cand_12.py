import numpy as np
def equation(t, x, v, params):
    return params[0]*np.exp(-params[1]*t) + params[2] + params[3]*v + params[4]*v**2 + params[5]*v**3 + params[6]*x + params[7]*x**3 + params[8]*t + params[9]*t**2
