import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*t+p[2]*x+p[3]*v+p[4]*x**2+p[5]*x**3+p[6]*v*np.abs(v)+p[7]*t*x
            +p[8]*t*t+p[9]*np.cos(t))