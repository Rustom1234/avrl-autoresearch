import numpy as np
def equation(t, x, v, params):
    e=np.exp(-params[6]*t)
    return (params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*v*t
            + e*(params[5] + params[7]*x + params[8]*v) + params[9]*t)
