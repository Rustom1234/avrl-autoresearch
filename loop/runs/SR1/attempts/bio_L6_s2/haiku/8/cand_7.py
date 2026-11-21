import numpy as np
def equation(t, P, params):
    g = np.exp(-params[4]*t)
    return P*(params[0] + params[1]*P + params[2]*P**2)*(1 + params[3]*t)*g
