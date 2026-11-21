import numpy as np
def equation(t, A, params):
    neg = -params[0]*A/(params[1]+A)/(1+params[2]*t)
    pos = params[3]*np.abs(A)**params[4]
    return neg + pos - params[5]*A
