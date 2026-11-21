import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*eps+p[2]*T+p[3]*eps*T+p[4]*eps**2+p[5]*T**2+p[6]*np.sqrt(eps)+p[7]*np.sqrt(eps)*T+p[8]*np.sqrt(eps)*T**2+p[9]*eps**2*T
