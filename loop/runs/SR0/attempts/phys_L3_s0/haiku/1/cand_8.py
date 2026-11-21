import numpy as np
def equation(t, x, v, params):
    p=params
    d=np.exp(-p[9]*t)
    return p[0] + p[1]*x + p[2]*v + p[3]*x**2 + p[4]*x**3 + p[5]*v**2 + p[6]*x*v + d*(p[7]*x + p[8]*v)