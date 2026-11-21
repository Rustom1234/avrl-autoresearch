import numpy as np
def equation(t, P, params):
    p=params
    return p[0]+p[1]*t+p[2]*t**2+p[3]*t**3+p[4]*P+p[5]*P**2+p[6]*P**3+p[7]*P*t+p[8]*P*t**2+p[9]*P**2*t
