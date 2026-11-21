import numpy as np
def equation(t, x, v, params):
    p = params
    return (p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*v + p[5]*v**2
            + p[6]*x*v + p[7]*x*v**2 + p[8]*x**2*v + p[9]*t)
