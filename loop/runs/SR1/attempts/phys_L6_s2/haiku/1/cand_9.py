import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x**params[2] + params[3]*v**params[4] + params[5]*t + params[6]*t**2 + params[7]*x*t + params[8]*v*t