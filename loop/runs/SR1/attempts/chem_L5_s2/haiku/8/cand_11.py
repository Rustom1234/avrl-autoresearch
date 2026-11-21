import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t**params[2] + params[3]*t + params[4]*A + params[5]*A*t
