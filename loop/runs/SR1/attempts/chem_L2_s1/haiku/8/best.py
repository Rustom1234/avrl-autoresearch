import numpy as np
def equation(t, A, params):
    return -(params[0]*A**params[1] + params[2]*A**params[3]*np.exp(params[4]*t) + params[5]*A**params[6]*np.exp(params[7]*t))
