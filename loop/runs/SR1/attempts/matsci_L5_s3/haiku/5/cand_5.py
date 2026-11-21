import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps + params[2]*T + params[3]*eps**2 + params[4]*T**2 + params[5]*eps*T + params[6]*eps**3 + params[7]*T**3 + params[8]*eps**2*T + params[9]*eps*T**2