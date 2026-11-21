import numpy as np
def equation(t, A, params):
    return -A**params[4]*np.exp(params[0]*A)*(params[1]*t**params[2]+params[3]*t**params[5])
