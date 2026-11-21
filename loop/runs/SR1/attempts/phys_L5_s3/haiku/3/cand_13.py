import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t**params[2] + params[3]*x + params[4]*x**2 + params[5]*x**3 + params[6]*np.sin(x) + params[7]*np.cos(x)
