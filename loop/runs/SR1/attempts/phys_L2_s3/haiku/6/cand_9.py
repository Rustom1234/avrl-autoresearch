import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*t+p[2]*x+p[3]*x**2+p[4]*x**3+p[5]*t*x+p[6]*t*x**2+p[7]*v+p[8]*t*x**3+p[9]*v*np.abs(v)
