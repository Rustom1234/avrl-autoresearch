import numpy as np
def equation(t, P, params):
    return params[0]*np.exp(-params[1]*P) + params[2]*t + params[3]*P + params[4]*P*t + params[5]*P*P + params[6]*t*P*P + params[7]