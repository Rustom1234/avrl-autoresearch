import numpy as np
def equation(t, P, params):
    g = 1 + params[4]*t*np.exp(-params[5]*t) + params[6]*np.sin(params[7]*t+params[8])
    return (params[0] + params[1]*P + params[2]*P**2 + params[3]*P**3) * g