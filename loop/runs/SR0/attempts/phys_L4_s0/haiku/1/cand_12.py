import numpy as np
def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*v + p[2]*v*t*t + p[3]*x + p[4]*x**2 + p[5]*x**3 + p[6]*x*t + p[7]/x + p[8]*v*v + p[9]*t*t