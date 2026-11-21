import numpy as np
def equation(t, A, params):
    p=params
    return -((p[0]+p[1]*t**p[2])*A + (p[3]+p[4]*t**p[5])*A**2 + p[6]*t**p[7])