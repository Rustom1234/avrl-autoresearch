import numpy as np
def equation(t, A, params):
    p=params
    return p[0]+p[1]*A+p[2]*A**2+p[3]*A**3+(p[4]+p[5]*A)*np.sin(p[6]*t+p[7])+p[8]*np.cos(p[6]*t)*(1.0)
