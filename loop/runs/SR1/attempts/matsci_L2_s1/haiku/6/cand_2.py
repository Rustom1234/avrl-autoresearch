import numpy as np
def equation(eps, T, params):
    s0 = params[0] + params[1]*T + params[2]*T**2
    Q = params[3] + params[4]*T
    b = params[5] + params[6]*T
    return -(s0 + Q*(1-np.exp(-b*eps)))