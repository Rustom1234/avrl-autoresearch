import numpy as np
def equation(t, P, params):
    e = np.exp(params[6]*t)
    return params[0] + params[1]*np.sqrt(P) + params[2]*P + params[3]*P**2 + params[4]*t + e*(params[5] + params[7]*np.sqrt(P) + params[8]*P)
