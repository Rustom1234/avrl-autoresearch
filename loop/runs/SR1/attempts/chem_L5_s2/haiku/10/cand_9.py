import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t**params[2] + params[3]*A + params[4]*A*t + params[5]*A*t**2
