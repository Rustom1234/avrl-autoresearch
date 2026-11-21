import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]+p[1]*T+p[2]*T**2+p[3]*T**3+p[4]*eps+p[5]*eps**2*T**2+p[6]*eps*T**2
