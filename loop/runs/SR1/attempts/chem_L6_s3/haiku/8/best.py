import numpy as np
def equation(t, A, params):
    E = np.exp(-params[3]*t)
    n = params[9]
    T0 = params[0] + params[1]*t + params[2]*E + params[4]*t*E
    T1 = params[5] + params[6]*t + params[7]*E + params[8]*t*E
    return T0 + T1*A**n