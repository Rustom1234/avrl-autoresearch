import numpy as np
def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3] + params[9]*t - params[4]*np.abs(v)**params[5] - params[6]*np.abs(x)**params[7] - params[8]*x**3
