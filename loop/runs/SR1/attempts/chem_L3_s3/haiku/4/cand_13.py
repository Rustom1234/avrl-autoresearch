import numpy as np
def equation(t, A, params):
    B = params[0] + params[1]*np.log(t) + params[2]*t + params[3]*t**3
    C = params[4] + params[5]*np.log(t) + params[6]*t + params[7]*t**3
    return B + C*A**2 + params[8]*A
