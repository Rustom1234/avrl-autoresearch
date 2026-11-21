import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x**3 + params[5]*v*x + params[6]*v**2 + params[7]*t + params[8]*np.sin(params[9]*t)