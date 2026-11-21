import numpy as np
def equation(t, A, params):
    p = params
    return (p[0]*A**2 + p[1]*t**2*A + p[2]*A**3 + p[3]*t**2*A**2 + p[4]*t**4/A**2
            + p[5]*A + p[6]*t/A + p[7]*t**3/A + p[8]*t**4/A + p[9]*t**2)
