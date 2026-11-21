import numpy as np
def equation(eps, T, params):
    a = params[0] + params[1]*T + params[2]*T**2
    b = params[3] + params[4]*T + params[5]*T**2
    c = params[6] + params[7]*T
    return a + b*eps**c
