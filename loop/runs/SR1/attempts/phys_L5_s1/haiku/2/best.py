import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*t+p[2]*t**2+p[3]*t**3+p[4]*x+p[5]*x**2+p[6]*x**3+p[7]*v+p[8]*v**2+p[9]*v**3
            )
