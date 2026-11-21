import numpy as np
def equation(t, x, v, params):
    return params[0] - params[1]*x - params[2]*x**3 + params[3]*np.cos(params[4]*t+params[5]) + (params[6] + params[7]*np.cos(params[8]*t+params[9]))*v
