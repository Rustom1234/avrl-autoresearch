import numpy as np
def equation(t, P, params):
    L=np.log(P)
    return params[0] + params[1]*P + params[2]*P**2 + params[3]*P*L + params[4]*t + params[5]*t*P + params[6]*t*P*L + params[7]*t*P**2 + params[8]*t**2 + params[9]*t**2*P