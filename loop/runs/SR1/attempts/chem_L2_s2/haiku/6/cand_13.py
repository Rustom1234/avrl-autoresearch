import numpy as np
def equation(t, A, p):
    g = 1 - np.exp(-p[7]*t**p[8])
    return p[0] + p[1]*A + p[2]*A**2 + p[3]*A**3 + (p[4] + p[5]*A + p[6]*A**2)*g + p[9]*t*A + (p[10]+p[11]*A)*np.exp(-p[12]*t)