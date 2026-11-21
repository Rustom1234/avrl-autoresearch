import numpy as np
def equation(t, A, params):
    A = np.abs(A)
    return (params[0]*np.exp(-params[1]*t) - params[2]*t**params[3]*A**params[4]
            + params[5]*A*np.exp(-params[6]*t) + params[7]*t*A + params[8]*t**2*A**2)
