import numpy as np
def equation(t, A, params):
    return -(params[0]+params[1]*t/(params[2]+t))*A*np.exp(-params[3]*A)
