import numpy as np
def equation(eps, T, params):
    p = params
    return -(p[0] + p[1]*T + p[2]*T**2 + p[3]*np.sqrt(T) + p[4]*np.exp(T) + p[5]*eps + p[6]*eps*np.sqrt(T) + p[7]*eps*T + p[8]*eps**2 + p[9]*np.log(eps))