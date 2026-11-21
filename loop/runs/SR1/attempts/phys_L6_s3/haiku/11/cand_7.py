import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*np.abs(x)**params[3] + params[4]*v + params[5]*np.abs(v)**params[6] + params[7]*np.cos(params[8]*t) + params[9]*np.sin(params[8]*t)
