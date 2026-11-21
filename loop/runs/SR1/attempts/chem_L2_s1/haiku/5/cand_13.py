import numpy as np
def equation(t, A, params):
    num = params[0]*A*(1+params[1]*t) + params[2]*A**2*(1+params[3]*t**2)
    den = 1 + params[4]*A*(1+params[5]*t)
    return -num/den