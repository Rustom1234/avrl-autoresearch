import numpy as np
def equation(t, A, params):
    p = params
    return (p[0]+p[1]*t+p[2]*t**2)/A*(-1) + (p[3]+p[4]*t) + (p[5]+p[6]*t)*A + p[7]*A**2
