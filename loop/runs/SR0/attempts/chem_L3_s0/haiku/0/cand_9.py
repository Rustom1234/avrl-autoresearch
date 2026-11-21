import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A**2 + params[3]*A**3 + params[4]*A*t/(params[5]+t) + params[6]*A*t
