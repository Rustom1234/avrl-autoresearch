import numpy as np
def equation(t, A, params):
    k = params[0]*np.exp(-params[1]*t)
    return -k*A**params[2] + params[3]*np.exp(-params[4]*t)*(params[5]-A)
