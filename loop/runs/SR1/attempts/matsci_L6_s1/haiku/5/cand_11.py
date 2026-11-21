import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]/T+p[2]*np.log(T)+p[3]*eps**p[4]+p[5]*eps**p[6]/T+p[7]*eps**p[8]*np.log(T)+p[9]*T*eps