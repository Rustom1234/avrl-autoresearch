import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x/v + params[2]/x + params[3]*np.sqrt(x) + params[4]*x + params[5]*np.exp(-x)