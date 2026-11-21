import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*v + p[2]*x + p[3]*x*t + p[4]*x**2 + p[5]*x**2*t + p[6]*x**3 + p[7]*t + p[8]*v**2 + p[9]*t**2