import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3+p[4]*v+p[5]*np.cos(p[6]*t)+p[7]*np.sin(p[6]*t)+p[8]*t+p[9]*t**2
