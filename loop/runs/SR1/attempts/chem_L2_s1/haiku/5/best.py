import numpy as np
def equation(t, A, params):
    num = params[0]*A + params[1]*A**2 + params[2]*t*A + params[3]*t*A**2
    den = 1 + params[4]*A + params[5]*t*A
    return -num/den