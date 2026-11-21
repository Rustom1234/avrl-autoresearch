import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.exp(-p[2]/T) + (p[3] + p[4]*np.exp(-p[5]/T))*(1-np.exp(-p[6]*eps)) + p[7]*eps + p[8]*eps**2 + p[9]*eps**3
