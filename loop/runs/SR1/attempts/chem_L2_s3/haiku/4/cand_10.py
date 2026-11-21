import numpy as np
def equation(t, A, params):
    return -(params[0] + params[1]*A) * (1 + params[2]*t)
