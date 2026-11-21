import numpy as np
def equation(t, x, v, params):
    return params[0]*t + params[1]*t**2 + params[2]