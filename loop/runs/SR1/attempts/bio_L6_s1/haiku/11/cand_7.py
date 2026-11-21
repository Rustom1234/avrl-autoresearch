import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P + params[3]*P**2 + params[4]*P**3 + params[5]*t*P + params[6]*t*P**2 + params[7]*P*np.log(P) + params[8]*t**2 + params[9]*t*P*np.log(P)