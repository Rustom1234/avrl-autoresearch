import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*v+p[3]*x**3+p[4]*v**2+p[5]*np.cos(p[6]*t)+p[7]*np.sin(p[6]*t)
