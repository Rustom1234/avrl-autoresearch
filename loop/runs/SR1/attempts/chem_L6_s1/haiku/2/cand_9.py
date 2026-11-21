import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1] + params[2]*A**params[3]*t + params[4]*A**params[5]*t**2 + params[6] + params[7]*t + params[8]/A
