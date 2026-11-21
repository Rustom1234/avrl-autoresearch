import numpy as np
def equation(t, A, params):
    return params[0]*t**params[2]/(params[1]+t**params[2]) + params[3]*A**params[4]*np.exp(-params[5]*t**params[6])
