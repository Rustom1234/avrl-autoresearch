import numpy as np
def equation(t, A, params):
    u = A*t
    return -params[0]*u**params[1]/(1+params[2]*u)