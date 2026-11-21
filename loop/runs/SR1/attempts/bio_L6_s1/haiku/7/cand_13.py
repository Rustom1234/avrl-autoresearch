import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*np.sqrt(P)*(1+params[2]*t**2) + params[3]*P + params[4]*P**2 + params[5]*t*P + params[6]*t
