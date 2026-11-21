import numpy as np
def equation(t, A, p):
    return -(p[0]*A**p[1]*t**p[2] + p[3]*A**p[4]*np.exp(p[5]*t))
