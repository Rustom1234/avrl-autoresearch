import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.exp(-p[2]*T) + p[3]*T + p[7]*T**2 + (p[4] + p[5]*T + p[6]*T**2)*eps + p[8]*eps**2
