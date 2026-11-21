import numpy as np
def equation(t, A, params):
    return -(params[0]*A**2 + params[1]*t**2)
