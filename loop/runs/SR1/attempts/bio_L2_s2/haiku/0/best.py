import numpy as np
def equation(t, P, params):
    p=params
    return (p[0]*P + p[1]*P**2 + p[2]*P**3
            + p[3]*t + p[4]*t**2 + p[5]*t**3
            + p[6]*P*t + p[7]*P*t**2 + p[8]*P**2*t + p[9]*P*t**3)
