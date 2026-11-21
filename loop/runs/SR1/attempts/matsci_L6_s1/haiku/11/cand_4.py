import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*T**p[2] + p[3]*eps**p[4]*T**p[5] + p[6]*eps**2 + p[7]*eps
