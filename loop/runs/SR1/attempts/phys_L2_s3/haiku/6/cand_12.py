import numpy as np
def equation(t, x, v, params):
    p=params
    s=np.sqrt(t)
    return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3+s*(p[4]+p[5]*x+p[6]*x**2+p[7]*x**3)+p[8]*v+p[9]*v*np.abs(v)
