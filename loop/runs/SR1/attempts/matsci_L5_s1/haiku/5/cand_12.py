import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T+p[2]*np.log(T)+p[3]/T+(p[4]+p[5]/T+p[6]*np.log(T))*eps**p[7]+p[8]*eps+p[9]*eps*np.log(T)
