import numpy as np
def equation(t, P, params):
    p=params
    return p[0]+p[1]*t+p[2]*t**2+(p[3]+p[4]*t)*P+p[5]*P**2+p[6]*P**3+p[7]*t*P**2+p[8]*t*P**3
