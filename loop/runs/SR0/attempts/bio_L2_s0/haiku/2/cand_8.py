import numpy as np
def equation(t, P, params):
    g = params[0]*(1-np.exp(-params[1]*t))
    return g + params[2]*P + params[3]*P**2 + params[4]*P*np.exp(-params[5]*t) + params[6]*P*np.log(P)*np.exp(-params[7]*t) + params[8]*t*P
