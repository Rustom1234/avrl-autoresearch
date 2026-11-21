import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*t + p[2]*v + p[3]*t*v + p[4]*t*t*v*v + p[5]*np.exp(-t) + p[6]*np.exp(v) + p[7]*np.exp(-v) + p[8]*x + p[9]*t*t
