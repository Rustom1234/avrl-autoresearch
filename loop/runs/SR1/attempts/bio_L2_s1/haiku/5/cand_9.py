import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + (params[2] + params[4]*t)*P*np.log(np.abs(params[3])/P) + params[5]*P
