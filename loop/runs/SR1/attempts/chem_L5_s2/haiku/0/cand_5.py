import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*t**3 + params[4]*A*t + params[5]*A + params[6]*A*t**2
