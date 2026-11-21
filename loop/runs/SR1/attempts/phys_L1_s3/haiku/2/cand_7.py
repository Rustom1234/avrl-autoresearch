import numpy as np
def equation(t, x, v, params):
    return params[0]*np.sin(params[1]*t+params[2]) + params[3] + params[4]/x + params[5]*x + params[6]*x*x + params[7]*v + params[8]*t
