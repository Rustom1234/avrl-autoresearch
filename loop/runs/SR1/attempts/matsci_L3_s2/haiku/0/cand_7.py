import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T + params[2]*T**2 + params[3]*eps**params[4] + params[5]*eps
            + params[6]*eps*T + params[7]*eps**2 + params[8]*eps**2*T + params[9]*T**3)
