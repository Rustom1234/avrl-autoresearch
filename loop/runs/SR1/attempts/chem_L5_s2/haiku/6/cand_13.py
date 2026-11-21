import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*np.sqrt(t) + params[4]*np.log(t) + params[5]*A
