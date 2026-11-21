import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*eps**params[3] + params[4]*eps*T + params[5]*T**2 + params[6]*eps**2
