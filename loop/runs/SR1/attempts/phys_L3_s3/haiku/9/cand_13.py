import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*x + p[2]*v + p[3]*x*x + p[4]*x*v + p[5]*v*v + p[6]*x**3
            + p[7]*x*x*v + p[8]*x*v*v + p[9]*v**3 + p[10]*np.cos(p[11]*t) + p[12]*np.sin(p[11]*t))
