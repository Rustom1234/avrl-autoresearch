import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P + params[3]*t*P + params[4]*np.exp(-params[5]*P) + params[6]*P*np.exp(-params[5]*P) + params[7]*t*np.exp(-params[5]*P) + params[8]*P*P
