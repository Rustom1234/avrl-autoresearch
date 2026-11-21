import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3+p[4]*x**4+p[5]*x**5+p[6]*v+p[7]*v*np.abs(v)+p[8]*t+p[9]*t**2
