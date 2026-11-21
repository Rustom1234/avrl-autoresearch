import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x**3 + params[2]*t/x + params[3]*np.sqrt(t) + params[4]*v
