import numpy as np
def equation(t, A, p):
    return p[0]*np.exp(-p[1]*t) + p[2]*A + p[3]*A**2 + p[4]
