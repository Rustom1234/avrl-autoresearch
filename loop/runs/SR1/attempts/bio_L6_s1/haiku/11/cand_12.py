import numpy as np
def equation(t, P, params):
    L=np.log(P)
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*P + params[4]*P*L + params[5]*P**2 + params[6]*t*P + params[7]*t*P*L + params[8]*t*P**2 + params[9]*P**2*L