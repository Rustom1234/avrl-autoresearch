import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]/t + params[3]*np.exp(-t)/t + params[4]*np.exp(-t) + (params[5] + params[6]*t)*A**2