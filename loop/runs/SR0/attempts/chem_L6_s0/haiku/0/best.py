import numpy as np
def equation(t, A, params):
    return params[0]*t*np.log(t) + params[1]*A**3/t + params[2]*A**1.5