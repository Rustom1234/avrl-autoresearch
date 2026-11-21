import numpy as np
def equation(eps, T, params):
    p = np.asarray(params)
    L = np.log(T)
    return (p[0] + p[1]*L + p[2]*eps + p[3]*eps*L + p[4]*eps**2 + p[5]*eps**2*L
            + p[6]*eps**3 + p[7]*eps**3*L + p[8]*eps*T + p[9]*eps**2*T)
