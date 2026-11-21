import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*t + p[2]*t*t + p[3]*v + p[4]*v*v + p[5]*v**3 + p[6]*x + p[7]*x*x + p[8]*np.exp(-t) + p[9]*t*v
