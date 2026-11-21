import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*A**3 + params[4]*A**3*t + params[5]*A*t + params[6]*A**2 + params[7]*A