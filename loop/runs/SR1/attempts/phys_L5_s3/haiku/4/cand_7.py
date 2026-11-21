import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*t**2 + params[4]*np.exp(-x) + params[5]*t