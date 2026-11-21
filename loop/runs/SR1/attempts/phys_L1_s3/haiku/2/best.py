import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*np.exp(params[3]*t) + params[4]/x + params[5]*x + params[6]*v + params[7]*t*t
