import numpy as np
def equation(t, A, params):
    return params[0]*t**params[1] + params[2]*A**params[3]*t**params[4] + params[5]*A**params[6]/t**params[7] + params[8]*A**params[9]