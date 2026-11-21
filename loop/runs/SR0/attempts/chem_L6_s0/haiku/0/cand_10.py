import numpy as np
def equation(t, A, params):
    return params[0]*t**params[1] - params[0]*t**params[2] + params[3]*A**3/t + params[4]*A**1.5