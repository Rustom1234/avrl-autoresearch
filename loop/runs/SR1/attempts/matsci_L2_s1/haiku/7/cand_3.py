import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*eps + p[2]*eps**2 + (p[3] + p[4]*eps)*T*np.exp(-p[5]*T) + p[6]*T + p[7]*T**2
