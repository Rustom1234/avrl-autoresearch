import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*t+p[2]*t**2+p[3]*x+p[4]*v+p[5]*x**3+p[6]*v*np.abs(v)+p[7]*x*v+p[8]*t*v+p[9]*t**2*v
