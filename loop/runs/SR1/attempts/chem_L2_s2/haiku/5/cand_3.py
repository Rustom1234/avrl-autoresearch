import numpy as np
def equation(t, A, p):
    return -p[0]*np.exp(-p[1]*t)*A**p[2] + p[3]*A**2 + p[4]*A + p[5]
