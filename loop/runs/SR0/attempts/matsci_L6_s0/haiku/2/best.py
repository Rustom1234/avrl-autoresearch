import numpy as np
def equation(eps, T, params):
    p=params
    return p[0]*eps**p[1] + p[2]*eps + p[3]*eps**2 + p[4]*T**3 + p[5]*eps*T + p[6]
