import numpy as np
def equation(t, x, v, params):
    return params[0]*np.log(t) + params[1]/x + params[2]
