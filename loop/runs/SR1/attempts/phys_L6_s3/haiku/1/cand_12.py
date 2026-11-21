import numpy as np
def equation(t, x, v, params):
    p=params
    return (-p[0]*v - p[1]*np.abs(v)**p[2]*np.sign(v) - p[3]*x - p[4]*np.abs(x)**p[5]*np.sign(x)
            + p[6]*np.cos(p[8]*t) + p[7]*np.sin(p[8]*t) + p[9])
