import numpy as np
def equation(t, P, params):
    r = params[0] + params[1]*np.cos(params[2]*t) + params[3]*np.sin(params[2]*t)
    return P*(r - params[4]*P)