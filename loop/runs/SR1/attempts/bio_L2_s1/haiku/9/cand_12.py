import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P + params[3]*t*P + params[4]*P*P + params[5]*np.exp(-params[6]*P)*(1+params[7]*t) + params[8]*t*P*P
