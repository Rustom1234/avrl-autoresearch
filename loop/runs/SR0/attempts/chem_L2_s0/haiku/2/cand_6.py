import numpy as np
def equation(t, A, params):
    p=params
    return p[0]*np.exp(-p[1]*t)*(p[2]/A + p[3]*A + p[4]) + p[5]*A + p[6]