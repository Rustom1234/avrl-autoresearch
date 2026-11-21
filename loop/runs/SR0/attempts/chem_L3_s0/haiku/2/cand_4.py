import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1] + params[2]*t*A**params[3] + params[4] + params[5]*t*A**2 + params[6]*A**2
