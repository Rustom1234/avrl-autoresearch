import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P**2 + params[3]*P**3 + params[4]*np.exp(-params[5]*P) + params[6]*np.exp(-params[7]*P)
