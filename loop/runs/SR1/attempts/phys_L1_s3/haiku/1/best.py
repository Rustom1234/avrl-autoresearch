import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t**params[2] + params[3]*x**params[4] + params[5]*v + params[6]*t*x**params[7]