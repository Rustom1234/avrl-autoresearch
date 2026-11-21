import numpy as np
def equation(eps, T, params):
    p=params
    a=p[0]+p[1]*T+p[2]*T**2
    b=p[3]+p[4]*T+p[5]*T**2
    return a + b*eps**p[6] + p[7]*np.log(eps) + p[8]*T*np.log(eps)+p[9]*T*eps
