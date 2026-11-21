import numpy as np
def equation(t, A, params):
    return params[0]*t - params[1]/(params[2]+t) + params[3]
