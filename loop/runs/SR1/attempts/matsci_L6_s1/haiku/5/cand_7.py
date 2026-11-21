import numpy as np
def equation(eps, T, params):
    L=np.log(T)
    p=params
    return p[0]+p[1]*L+p[2]*eps+p[3]*eps**2+p[4]*eps*L+p[5]*eps**2*L+p[6]*L**2+p[7]*eps**3+p[8]*eps**3*L+p[9]*eps*L**2