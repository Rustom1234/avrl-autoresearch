import numpy as np
def equation(t, P, params):
    return params[0]*P*np.log(params[1]/P) + params[2] + params[3]*t + params[4]*t*P + params[5]*t**2 + params[6]*P**2
