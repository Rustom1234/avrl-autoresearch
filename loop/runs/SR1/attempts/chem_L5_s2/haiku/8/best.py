import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]/t + params[4]*np.sqrt(t) + params[5]*np.exp(-t/2) + params[6]*A + params[7]*np.log(t) + params[8]*t*A
