import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]*A**2) * (1 - np.exp(-params[2]*t)) + params[3]*A
