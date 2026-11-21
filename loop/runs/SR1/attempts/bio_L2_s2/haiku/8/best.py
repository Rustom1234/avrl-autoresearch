import numpy as np
def equation(t, P, params):
    p=params
    one=np.ones_like(t)
    base = (p[0]+p[1]*t+p[2]*t**2+p[3]*t**3+p[4]*t**4)*one \
         + (p[5]+p[6]*t+p[7]*t**2+p[8]*t**3+p[9]*t**4)*P
    return base + 0*P
