import numpy as np
def equation(t, x, v, params):
    F = params[0]*np.cos(params[1]*t+params[2])
    return F + params[3] - params[4]*np.tanh(params[5]*v) - params[6]*v - params[7]*x - params[8]*x**2 - params[9]*x**3
