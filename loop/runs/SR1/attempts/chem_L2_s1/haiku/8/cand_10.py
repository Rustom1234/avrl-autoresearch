import numpy as np
def equation(t, A, params):
    return -params[0]*A**params[1]*(1+params[2]*t**params[3])/(1+params[4]*A**params[5]*t**params[6])
