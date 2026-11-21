import numpy as np
def equation(t, A, params):
    p = params[1]+params[2]*t
    b = params[3]+params[4]*t
    return -params[0]*A**p*np.exp(-b*A)
