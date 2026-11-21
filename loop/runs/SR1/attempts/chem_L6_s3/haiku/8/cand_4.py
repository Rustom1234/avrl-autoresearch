import numpy as np
def equation(t, A, params):
    g = params[0] + params[1]*t + params[2]*t**2
    h = params[3] + params[4]*t + params[5]*t**2
    return g + h*A**params[6]