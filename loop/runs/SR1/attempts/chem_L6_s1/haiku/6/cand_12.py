import numpy as np
def equation(t, A, params):
    p=params[6]; q=params[7]
    return (params[0] + params[1]*t + params[2]*t**2)*A**p + (params[3] + params[4]*t + params[5]*t**2)*A**q
