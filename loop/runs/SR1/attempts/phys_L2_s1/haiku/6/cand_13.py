import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*x*v + p[2]*np.cos(x) + p[3]*np.exp(-x) + p[4]/v + p[5]*x + p[6]*v + p[7]*x**3 + p[8]*v*v + p[9]*x*x
