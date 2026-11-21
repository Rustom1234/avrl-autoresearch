import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*np.log(eps)+p[2]*eps**p[3]+(p[4]+p[5]*np.log(eps)+p[6]*eps**p[7])*T+p[8]*T**2
