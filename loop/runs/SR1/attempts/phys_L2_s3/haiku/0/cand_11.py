
import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*t + p[2]*x**p[3] + p[4]*v + p[5]*np.abs(v)*v
            + p[6]*np.cos(p[7]*t) + p[8]*np.sin(p[7]*t) + p[9]*x*v)
