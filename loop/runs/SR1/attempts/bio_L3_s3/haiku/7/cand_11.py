import numpy as np
def equation(t, P, params):
    g = t*np.exp(-params[6]*t)
    return params[0] + params[1]*P + params[2]*P**2 + (params[3] + params[4]*P + params[5]*P**2)*g + params[7]*np.log(t)
