import numpy as np
def equation(t, A, params):
    p = params
    return -(p[0] + p[1]*A + p[2]*A**2 + p[3]*A**3 + p[4]*t + p[5]*t*A + p[6]*t*A**2 + p[7]*t**2 + p[8]*t**2*A + p[9]*t**3)
