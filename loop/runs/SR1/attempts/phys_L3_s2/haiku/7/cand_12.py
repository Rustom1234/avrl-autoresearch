import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*x+p[2]*v+p[3]*x**2+p[4]*x*v+p[5]*x**3+p[6]*x**2*v+p[7]*np.cos(p[9]*t)+p[8]*np.sin(p[9]*t))
