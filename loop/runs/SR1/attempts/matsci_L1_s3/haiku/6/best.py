import numpy as np
def equation(eps, T, params):
    s0 = params[0]+params[1]*T+params[2]*T**2+params[3]*T**3
    K = params[4]
    return -(s0 + K*eps**params[5])