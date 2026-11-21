import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps**params[2] + (params[3] + params[4]*eps**params[5])/T**params[6] + params[7]*T + params[8]*eps*T**2 + params[9]*eps**3*T
