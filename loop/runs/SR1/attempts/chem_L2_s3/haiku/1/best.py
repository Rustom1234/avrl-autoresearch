import numpy as np
def equation(t, A, params):
    return -(params[0]/t + params[1]*A**params[2] + params[3]*t*A**params[4] + params[5] + params[6]*np.exp(-params[7]*t))
