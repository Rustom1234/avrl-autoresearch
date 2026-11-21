import numpy as np
def equation(t, x, v, params):
    p=params
    return -p[0] - p[1]*x - p[2]*x**3 - p[3]*v - (p[4] + p[5]*x + p[6]*x**2)/v - p[7]*x**2 - p[8]*v**2
