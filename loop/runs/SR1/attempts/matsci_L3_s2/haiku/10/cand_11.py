import numpy as np
def equation(eps, T, params):
    p=params
    F = np.tanh(p[2]*T)
    return p[0] + p[1]*F*np.exp(-p[3]*eps) + p[4]*F*eps + p[5]*eps + p[6]*eps**2 + p[7]*np.exp(-p[8]*eps)*F*eps + p[9]*T
