import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*np.cos(params[2]*t+params[3]) + params[4]*x + params[5]*x**3 + params[6]*v + params[7]*np.tanh(v) + params[8]*t**2*np.tanh(v) + params[9]*t**2
