import numpy as np
def equation(t, A, params):
    k = params[0]*t**params[1]
    return -k*A/(1+params[2]*A)
