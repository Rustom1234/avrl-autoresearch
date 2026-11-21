import numpy as np
def equation(t, x, v, params):
    p=params
    poly = p[0] + p[1]*x + p[2]*v + p[3]*x**2 + p[4]*x*v + p[5]*v**2 + p[6]*x**3
    g = np.exp(-p[7]*t)*(p[8]*np.cos(p[9]*t))
    return poly + g
