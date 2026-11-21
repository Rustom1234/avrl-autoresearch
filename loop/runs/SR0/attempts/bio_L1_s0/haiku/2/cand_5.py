import numpy as np
def equation(t, P, params):
    return params[0]/P + params[1] + params[2]*P + params[3]*P**2 + params[4]*P*np.log(P) + params[5]*np.log(P) + params[6]*t*P + params[7]*t/P
