import numpy as np
def equation(eps, T, params):
    a = params[0] + params[1]*eps + params[2]*eps**2
    b = params[3] + params[4]*eps + params[5]*eps**2
    return a + b*T**params[6]
