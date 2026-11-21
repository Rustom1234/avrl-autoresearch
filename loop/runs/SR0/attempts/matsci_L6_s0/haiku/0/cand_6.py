import numpy as np
def equation(eps, T, params):
    return params[0]*eps**params[1] + params[2]*T*eps**params[3] + params[4]*T**2 + params[5]*T + params[6]*eps*T**2
