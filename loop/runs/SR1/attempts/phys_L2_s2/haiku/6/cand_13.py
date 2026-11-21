import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*x**2+p[3]/x+p[4]*x/v+p[5]/x**2+p[6]*x**3+p[7]*x**2/v+p[8]*x/v**2+p[9]*x*v