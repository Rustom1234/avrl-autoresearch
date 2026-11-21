import numpy as np
def equation(t, P, params):
    p=params
    return p[0]+p[1]*P+p[2]*P**2+(p[3]+p[4]*P)*np.sin(p[5]*t)+(p[6]+p[7]*P)*np.cos(p[5]*t)+p[8]*t
