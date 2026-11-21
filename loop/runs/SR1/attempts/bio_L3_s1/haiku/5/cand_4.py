import numpy as np
def equation(t, P, params):
    return params[0]*P + params[1]*P**2 + params[2]/P + params[3] + params[4]*t + params[5]*t*P + params[6]*np.sin(params[7]*t+params[8])
