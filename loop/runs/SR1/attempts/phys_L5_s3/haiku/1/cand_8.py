import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*np.cos(params[2]*t+params[3]) + params[4]*x + params[5]*v + params[6]*x**params[7] + params[8]*v*np.abs(v)**params[9]
