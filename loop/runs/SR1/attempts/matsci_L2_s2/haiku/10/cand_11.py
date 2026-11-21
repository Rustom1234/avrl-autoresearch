import numpy as np
def equation(eps, T, params):
    p=params
    L=np.log(eps)
    base = p[0]/eps + p[1]*L + p[2]*eps + p[3]*eps**2 + p[4]
    Tc = p[5]*T + p[6]*T**2
    return base + Tc*(1/eps) + p[7]*T*L + p[8]*T*eps + p[9]*T*eps**2
