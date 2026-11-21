import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*v+p[3]*v*v+p[4]*v**3+p[5]*x*x+p[6]*x**3+p[7]*t+p[8]*np.cos(p[9]*t)+p[10]*np.sin(p[9]*t)
