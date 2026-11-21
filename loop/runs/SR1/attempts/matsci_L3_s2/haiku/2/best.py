import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.tanh(p[2]*T) + p[3]*(1-np.exp(-p[4]*eps)) + p[5]*eps + p[6]*eps*np.exp(-p[7]*eps)*np.tanh(p[8]*T)
