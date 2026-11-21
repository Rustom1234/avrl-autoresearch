import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*x+p[2]*x**2+p[3]*x**3+p[4]*v+p[5]*v**2+p[6]*v**3
            +p[7]*x*v+p[8]*np.cos(p[9]*t))
