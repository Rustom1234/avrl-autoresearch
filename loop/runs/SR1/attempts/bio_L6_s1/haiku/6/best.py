import numpy as np
def equation(t, P, params):
    L = np.log(P)
    return (params[0] + params[1]*L + params[2]*t + params[3]*t*L + params[4]*P + params[5]*P**2
            + params[6]*P*t + params[7]*t**2 + params[8]*t**2*L + params[9]*P**2*t)
