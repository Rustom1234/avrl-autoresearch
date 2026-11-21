import numpy as np
def equation(t, A, params):
    T=[np.ones_like(t), t, 1/t, np.sqrt(t)]
    p=params
    out = p[0]*T[0] + p[1]*T[1] + p[2]*T[2] + p[3]*T[3]
    out = out + (p[4]*T[0] + p[5]*T[1] + p[6]*T[2] + p[7]*T[3])*A**2
    out = out + (p[8]*T[2] + p[9]*T[3])*A
    return -out
