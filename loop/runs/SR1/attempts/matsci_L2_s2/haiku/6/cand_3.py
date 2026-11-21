import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*np.log(eps)+p[2]*eps+p[3]*eps**2+p[4]*T+p[5]*T**2+p[6]*T*np.log(eps)+p[7]*T*eps+p[8]*np.log(eps)*eps+p[9]*T**2*np.log(eps)
