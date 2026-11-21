import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x/v + params[3]*np.sqrt(x) + params[4]*x**2 + params[5]/x
