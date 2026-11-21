import numpy as np
def equation(t, x, v, params):
    return params[0]*np.exp(-params[1]*t)*np.cos(params[2]*t+params[3]) + params[4] + params[5]*x + params[6]*v + params[7]*x**3 + params[8]*x**2
