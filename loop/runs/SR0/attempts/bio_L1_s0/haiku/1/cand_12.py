import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P**2 + params[3]/P + params[4]*P*np.log(P) + params[5]*np.log(P) + params[6]*P**3
