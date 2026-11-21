import numpy as np
def equation(t, A, params):
    return -params[0]*np.exp(-params[1]*t)*A**params[2]*(1-(A/params[3])**params[4])
