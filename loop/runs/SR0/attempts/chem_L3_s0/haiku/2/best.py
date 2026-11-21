import numpy as np
def equation(t, A, params):
    return (params[0]+params[1]*t**params[2])*A + (params[3]+params[4]*t**params[5])*A**2 + params[6]*A**3 + params[7] + params[8]*t*A**3 + params[9]*t
