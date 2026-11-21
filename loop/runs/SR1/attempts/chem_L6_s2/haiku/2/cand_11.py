import numpy as np
def equation(t, A, params):
    return -params[0]*A**params[1] + params[2]*t**2/A + params[3]/A**params[4]
