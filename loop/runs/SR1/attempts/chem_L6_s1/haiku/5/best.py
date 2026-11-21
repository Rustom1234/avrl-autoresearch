
import numpy as np
def equation(t, A, params):
    return params[0]*A**params[1]*np.cos(params[2]*t+params[3]) + params[4]*A**params[5] + params[6]*t*A**params[7]
