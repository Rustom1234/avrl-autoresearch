import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]*np.exp(-params[2]*t) + params[3]*t**3*A**2
            + params[4]*t**2*A + params[5]*t**2*np.log(A) + params[6]/t
            + params[7]*t*np.exp(-t))
