import numpy as np
def equation(t, A, p):
    h = p[0]*(1-np.exp(-p[1]*t)) + p[2] + p[6]*t
    g = p[3]*np.exp(-p[4]*t)
    return h + g*A**p[5]
