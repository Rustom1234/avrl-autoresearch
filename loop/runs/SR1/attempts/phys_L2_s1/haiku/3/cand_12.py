import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*x**3+p[3]*v+p[4]*v**3+p[5]*x*v+p[6]*x**2*v+p[7]*x*v**2+p[8]*x**2+p[9]*v**2