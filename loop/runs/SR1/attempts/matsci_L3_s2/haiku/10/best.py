import numpy as np
def equation(eps, T, params):
    p=params
    F = np.tanh(p[2]*T)
    return p[0] + p[1]*F + p[3]*eps**p[4]*F + p[5]*eps**p[6] + p[7]*np.exp(-p[8]*eps)*F + p[9]*T*eps
