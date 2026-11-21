import numpy as np
def equation(t, A, params):
    e = np.exp(-params[2]*t)
    return params[0] + params[1]*e + (params[3] + params[4]*e)*A**0.75 + (params[5] + params[6]*e)*np.sqrt(A)
