import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A**2 + params[3]/A + (params[4] + params[5]/A + params[6]*A)*t**params[7] + params[8]*t**2
