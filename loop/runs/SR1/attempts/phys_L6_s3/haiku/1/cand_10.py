import numpy as np
def equation(t, x, v, params):
    p=params
    return (-p[0]*v - p[4]*v*np.abs(v) - p[1]*x - p[3]*x**2 - p[2]*x**3
            + p[5]*np.cos(p[7]*t) + p[6]*np.sin(p[7]*t) + p[8])
