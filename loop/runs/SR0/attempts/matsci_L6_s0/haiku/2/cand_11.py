import numpy as np
def equation(eps, T, params):
    return params[0]*eps**params[1] - params[2]*T**3 - params[3]*eps*T
