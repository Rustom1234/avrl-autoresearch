import numpy as np
def equation(t, x, v, params):
    F = params[0]*np.exp(-params[1]*t)*np.cos(params[2]*t+params[3])
    return F - params[4]*v - params[5]*x - params[6]*x**3 - params[7]*v*np.abs(v) + params[8]