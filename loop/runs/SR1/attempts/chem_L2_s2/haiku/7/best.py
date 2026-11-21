import numpy as np
def equation(t, A, params):
    k = params[0]*np.exp(-params[1]*t)
    return (params[2]*A + params[3]*A**2 + params[4]*A**3)/(1 + params[5]*A) + k*A*A + params[6] + params[7]*np.exp(-params[8]*t)*A