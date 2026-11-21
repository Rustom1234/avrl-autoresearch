import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*np.log(x) + params[2]/x + params[3]*x/v
