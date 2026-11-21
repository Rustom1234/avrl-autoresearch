import numpy as np
def equation(t, A, params):
    return (params[0]*A**params[1] + params[2]*t*A**params[3]) / (1 + params[4]*t)
