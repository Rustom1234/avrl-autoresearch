import numpy as np
def equation(eps, T, params):
    p=params
    n=p[2]+p[3]*T
    return p[0]+p[1]*eps**n + p[4]*np.log(eps) + p[5]*T*np.log(eps) + p[6]*T + p[7]*eps + p[8]*T*eps + p[9]*T**2
