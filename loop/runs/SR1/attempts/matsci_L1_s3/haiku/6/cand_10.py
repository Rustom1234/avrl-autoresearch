import numpy as np
def equation(eps, T, params):
    s0 = params[0]+params[1]*T+params[2]*T**2
    K = params[3]+params[4]*T+params[5]*T**2
    return -(s0 + K*eps**params[6])