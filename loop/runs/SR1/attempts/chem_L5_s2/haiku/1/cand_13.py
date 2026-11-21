import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*np.log(t) + params[2]*t + params[3]/t + params[4]/t**2
