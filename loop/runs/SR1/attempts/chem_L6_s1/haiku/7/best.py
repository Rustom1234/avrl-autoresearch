import numpy as np
def equation(t, A, params):
    p=params
    return (p[0]+p[1]*t+p[2]*t**2+p[3]*np.exp(-t))*(1+0*A) + (p[4]+p[5]*t+p[6]*t**2)*A + p[7]*A**2 + (p[8]+p[9]*t)/A**2