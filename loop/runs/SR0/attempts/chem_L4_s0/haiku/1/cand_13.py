import numpy as np
def equation(t, A, params):
    p=params
    return (p[0]*A + p[1]*A**2 + p[2]*t*A + p[3]*t*A**2 + p[4]*t**2*A + p[5]*A**3
            + p[6]*A/(1+A) + p[7]*t*A/(1+A) + p[8]*t + p[9]/A)
