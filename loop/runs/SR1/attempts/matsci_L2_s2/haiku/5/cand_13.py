import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.exp(-p[2]*eps) + p[3]*np.cos(p[4]*eps*T) + p[5]*np.log(eps)/T + p[6]*eps*T + p[7]*np.sin(p[4]*eps*T)