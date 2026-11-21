import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*x**3 + params[3]*v
            + params[4]*np.tanh(params[5]*v) + params[6]*np.cos(params[7]*t)
            + params[8]*np.sin(params[7]*t) + params[9]*t)