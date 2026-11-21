import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.tanh(p[2]*T) + (p[3] + p[4]*np.tanh(p[5]*T))*(1-np.exp(-p[6]*eps)) + p[7]*eps + p[8]/eps + p[9]*np.sqrt(eps)
