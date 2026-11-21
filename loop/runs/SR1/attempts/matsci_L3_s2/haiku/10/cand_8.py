import numpy as np
def equation(eps, T, params):
    p=params
    g = tanh_ = np.tanh(p[2]*T)
    return p[0] + p[1]*np.tanh(p[2]*T) + p[3]*eps**p[4] + p[5]*np.exp(-p[6]*eps) + p[7]*np.tanh(p[2]*T)*eps + p[8]*eps**2*np.tanh(p[2]*T)
