import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*np.log(T)+p[2]*eps**p[3]*T**p[4]+p[5]*eps**p[6]*np.log(T)