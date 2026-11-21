import numpy as np
def equation(t, x, v, params):
    return (params[0]*np.cos(params[1]*t+params[2]) + params[3] + params[4]*v + params[5]*v*np.abs(v) + params[6]*v**3
            + params[7]*x + params[8]*x**3 + params[9]*x*v)