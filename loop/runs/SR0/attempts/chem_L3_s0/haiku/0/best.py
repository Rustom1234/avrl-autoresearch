import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1] - params[2] + params[3]*A*t/(params[4]+t) + params[5]*A**2
