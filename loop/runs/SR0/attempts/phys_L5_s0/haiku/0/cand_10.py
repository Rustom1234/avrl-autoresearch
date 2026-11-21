import numpy as np
def equation(t, x, v, params):
    p=params
    static = p[0] + p[1]*x + p[2]*v + p[3]*x**3 + p[4]*v**2
    env = np.exp(-p[5]*t)
    return static + env*(p[6] + p[7]*x + p[8]*v + p[9]*x*v)
