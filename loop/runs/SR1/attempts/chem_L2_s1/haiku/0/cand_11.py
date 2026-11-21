import numpy as np
def equation(t, A, params):
    p=params
    return -(p[0]*A + p[1]*A**2 + p[2]*A**3 + (p[3]+p[4]*A)*np.exp(p[5]*t) + (p[6]+p[7]*A)*t**3)