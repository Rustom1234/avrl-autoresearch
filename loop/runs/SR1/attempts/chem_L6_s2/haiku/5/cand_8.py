import numpy as np
def equation(t, A, params):
    return -params[0]/A**params[1] - params[2]*A**params[3] + params[4]*t/A**params[5] + params[6]*t