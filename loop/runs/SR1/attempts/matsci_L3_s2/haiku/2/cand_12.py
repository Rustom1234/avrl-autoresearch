import numpy as np
def equation(eps, T, params):
    p=params
    return p[0] + p[1]*T**p[2] + (p[3] + p[4]*T**p[5])*eps**p[6] + (p[7] + p[8]*T)/eps + p[9]*eps
