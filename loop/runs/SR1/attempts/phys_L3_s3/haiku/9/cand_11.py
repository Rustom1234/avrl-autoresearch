import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*x + p[2]*x**3 + p[3]*v + p[4]*v*np.abs(v)**p[5]
            + p[6]*np.cos(p[7]*t) + p[8]*np.sin(p[7]*t) + p[9]*t*t)
