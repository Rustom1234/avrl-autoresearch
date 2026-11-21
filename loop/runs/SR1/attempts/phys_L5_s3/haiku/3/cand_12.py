import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t**params[2] + params[3]*np.sin(params[4]*x) + params[5]*x + params[6]*x**2 + params[7]*x**3 + params[8]*np.cos(params[9]*x)
