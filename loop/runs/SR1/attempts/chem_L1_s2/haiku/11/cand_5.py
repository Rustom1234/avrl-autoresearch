import numpy as np
def equation(t, A, params):
    return -params[0]*t**params[1]*A**params[2]/(1+params[3]*A*t**params[4])