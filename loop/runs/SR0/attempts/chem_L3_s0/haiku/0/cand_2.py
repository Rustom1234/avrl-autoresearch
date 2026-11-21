import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]*t)*A**params[2] - params[3]*A - params[4]*A**2
