import numpy as np
def equation(eps, T, params):
    p=params
    a=p[0]+p[1]*T+p[2]*T**2
    return a + p[3]*np.log(eps) + p[4]*(1-np.exp(-p[5]*eps)) + p[6]*T*np.log(eps) + p[7]*T*(1-np.exp(-p[8]*eps)) + p[9]*eps
