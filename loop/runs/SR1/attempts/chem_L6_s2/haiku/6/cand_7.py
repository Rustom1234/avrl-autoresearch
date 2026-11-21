import numpy as np
def equation(t, A, params):
    a = params[0] + params[1]*t
    b = params[2] + params[3]*t
    return -a/A - b*A**params[4]
