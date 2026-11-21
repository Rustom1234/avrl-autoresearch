import numpy as np
def equation(eps, T, params):
    p=params
    return (p[0]+p[1]*T)*eps**p[2] + (p[3]+p[4]*T)*np.log(eps) + p[5] + p[6]*T + p[7]*eps + p[8]*T*eps + p[9]*T**2
