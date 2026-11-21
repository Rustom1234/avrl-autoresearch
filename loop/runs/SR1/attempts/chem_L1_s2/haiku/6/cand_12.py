import numpy as np
def equation(t, A, params):
    return -(params[0]*t**2 + params[1]*A**2)