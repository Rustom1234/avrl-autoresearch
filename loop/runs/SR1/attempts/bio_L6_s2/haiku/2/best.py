import numpy as np
def equation(t, P, params):
    return -(params[0] + params[1]*np.exp(params[2]*t)) * P**1.5
