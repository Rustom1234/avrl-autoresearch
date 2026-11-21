import numpy as np
def equation(eps, T, params):
    p=params
    base = p[0] + p[1]/T + p[2]*T + p[3]*T**2
    g = p[4] + p[5]*T + p[6]*T**2 + p[7]/T
    return base + eps*g + p[8]*eps**2 + p[9]*eps**2*T
