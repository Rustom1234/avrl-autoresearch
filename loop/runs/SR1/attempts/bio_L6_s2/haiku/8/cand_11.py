import numpy as np
def equation(t, P, params):
    return P*(params[0] + params[1]*np.exp(-params[2]*t)) + P**2*(params[3] + params[4]*np.exp(-params[5]*t)) + params[6] + params[7]*t
