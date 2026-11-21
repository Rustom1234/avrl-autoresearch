import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A*np.log(A) + (params[3] + params[4]*A)*t**3
