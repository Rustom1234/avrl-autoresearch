import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]*np.exp(-params[2]*t))/t + params[3] + params[4]*t + params[5]*t**2 + (params[6] + params[7]*t)*A**2