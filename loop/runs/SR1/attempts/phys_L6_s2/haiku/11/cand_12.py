import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*t*np.exp(-params[2]*x) + params[3]*x + params[4]*x**2 + params[5]*x**3
            + params[6]*v + params[7]*v**2 + params[8]*v*x + params[9]*t)