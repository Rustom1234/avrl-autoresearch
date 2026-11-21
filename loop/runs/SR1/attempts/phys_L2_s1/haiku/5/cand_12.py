import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*v+p[2]*v**2+p[3]*v**3+p[4]*x+p[5]*x**2+p[6]*x**3
            +p[7]*x*v+p[8]*x*x*v+p[9]*x*v*v)