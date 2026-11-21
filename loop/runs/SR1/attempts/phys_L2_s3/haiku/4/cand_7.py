import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*np.abs(x)**params[3]*np.sign(x) + params[4]*np.sign(v)*np.abs(v)**params[5] + params[6]*v + params[7]*t + params[8]*t**2 + params[9]*x*v
