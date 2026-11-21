import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*x + params[4]*v + params[5]*x**3 + params[6]*np.exp(-params[7]*t) + params[8]/v + params[9]*np.exp(-x)
