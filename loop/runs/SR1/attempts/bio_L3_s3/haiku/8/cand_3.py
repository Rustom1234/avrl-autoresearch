import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*np.sin(params[3]*t + params[4]) + params[5]*P*np.sin(params[3]*t+params[4])
