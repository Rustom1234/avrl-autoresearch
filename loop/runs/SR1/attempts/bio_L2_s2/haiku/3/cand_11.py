import numpy as np
def equation(t, P, params):
    return P*(params[0] + params[1]*t + params[2]*t**2) - P**2*(params[3] + params[4]*t) + params[5]*np.exp(params[6]*t) + params[7] + params[8]*t
