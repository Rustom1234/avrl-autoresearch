import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T+p[2]*eps+p[3]*eps**2+p[4]*eps*T+p[5]*eps**2*T+p[6]*eps*T**2+p[7]*T**2+p[8]*eps**3+p[9]*T**3
