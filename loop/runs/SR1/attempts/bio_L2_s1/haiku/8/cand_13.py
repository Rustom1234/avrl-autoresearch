import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]/P + params[3]*np.sqrt(P) + params[4]*np.exp(-P) + params[5]*t*np.sqrt(P) + params[6]*P*np.log(P)
