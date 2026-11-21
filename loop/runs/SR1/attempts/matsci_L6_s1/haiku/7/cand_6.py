import numpy as np
def equation(eps, T, params):
    s0 = params[0] + params[1]*T + params[2]*T**2 + params[3]*T**3
    Q = params[4] + params[5]*T + params[6]*T**2
    return s0 + Q*(1-np.exp(-params[7]*eps)) + params[8]*eps*T + params[9]*eps**2