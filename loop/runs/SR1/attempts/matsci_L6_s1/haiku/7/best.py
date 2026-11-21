import numpy as np
def equation(eps, T, params):
    p = params
    e0 = p[0] + p[1]*T + p[2]*T**2 + p[3]/T
    e1 = p[4] + p[5]/T
    e2 = p[6] + p[7]/T
    e3 = p[8]/T
    return e0 + eps*e1 + eps**2*e2 + eps**3*e3