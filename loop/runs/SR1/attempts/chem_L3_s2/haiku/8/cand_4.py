import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]*A**params[2]) * (1 - np.exp(-params[3]*t)) / (1 + params[4]*t)
