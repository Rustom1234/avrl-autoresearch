import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3+p[4]*v+p[5]*v*np.abs(v)+p[6]*t+p[7]*t**2+p[8]*np.cos(p[9]*t)
