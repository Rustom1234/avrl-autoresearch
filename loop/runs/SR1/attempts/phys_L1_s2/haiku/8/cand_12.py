import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*np.sqrt(t)+p[2]*x+p[3]*x**2+p[4]*x**3+p[5]*v+p[6]*v*np.abs(v)+p[7]*x*v+p[8]*np.sin(p[9]*x)
