import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.tanh(p[2]*T) + p[3]*np.exp(-p[4]*eps) + p[5]*eps + p[6]*eps**2 + p[7]*np.tanh(p[2]*T)*np.exp(-p[4]*eps)
