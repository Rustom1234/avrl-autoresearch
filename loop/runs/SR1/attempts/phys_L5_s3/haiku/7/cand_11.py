import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x*np.exp(-p[2]*x)+p[3]*x+p[4]*v+p[5]*v*np.abs(v)+p[6]*t+p[7]*t**2+p[8]*np.cos(t)+p[9]*np.sin(t)
