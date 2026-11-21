import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*v**2 + params[5]*v*np.exp(-params[6]*v) + params[7]*v*x
            + params[8]*np.cos(params[9]*t) + params[8]*0*t + 0*x)
