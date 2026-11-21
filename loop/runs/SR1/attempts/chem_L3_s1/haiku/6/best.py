import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*A + params[3]*A**2 + params[4]*t*A + params[5]*t*A**2 + params[6]*A**3
