import numpy as np
def equation(t, P, params):
    return np.exp(params[0] + params[1]*t) * P**(params[2] - params[3]*t)
