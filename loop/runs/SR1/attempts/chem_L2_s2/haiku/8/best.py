import numpy as np
def equation(t, A, params):
    return (params[0]*np.exp(-params[1]*t) + params[2]*A*np.exp(-params[3]*t) + params[4]*A**2 + params[5]*A**3*np.exp(-params[6]*t) + params[7]*np.exp(-params[8]*t)*A**2 + params[9])