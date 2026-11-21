import numpy as np
def equation(eps, T, params):
    p=params
    L=np.log(eps)
    s=np.sqrt(T)
    return (p[0]+p[1]*L+p[2]*eps)*(1+p[3]*s+p[4]*T) + p[5]*eps*L*s + p[6]*eps**2*L + p[7]*eps**2 + p[8]*T*L + p[9]*eps*T
