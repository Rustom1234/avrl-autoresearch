import numpy as np
def equation(t, A, p):
    return p[0]*A**p[1]*t**p[2]*np.exp(-p[3]*t) + p[4]*A**p[5]
