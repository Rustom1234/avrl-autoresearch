import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1] - params[2]*A**params[3]*t**params[4] + params[5]*t**params[6]/A**params[7]
