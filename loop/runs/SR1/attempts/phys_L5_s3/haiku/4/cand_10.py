import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*x*np.exp(-params[4]*x) + params[5]*x + params[6]*x**2 + params[7]*np.exp(-x)