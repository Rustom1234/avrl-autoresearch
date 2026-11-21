import numpy as np
def equation(t, P, params):
    return params[0]*P*np.log(params[1]/P) + params[2]/P + params[3]*P + params[4]*P**2 + params[5]
