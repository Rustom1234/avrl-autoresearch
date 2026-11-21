import numpy as np
def equation(t, A, p):
    return p[0]*np.exp(-p[1]*t) + p[2]*A*np.exp(-p[3]*t) + p[4]*A + p[5]*t + p[6]*A*t + p[7]
