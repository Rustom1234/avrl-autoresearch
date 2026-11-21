import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*v + p[2]*v**2 + p[3]*v**3
            + p[4]*np.exp(-p[5]*t) + p[6]*v*np.exp(-p[7]*t) + p[8]*np.exp(-p[9]*t)*np.cos(0.0*t))