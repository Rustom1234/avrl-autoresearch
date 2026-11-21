import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P*P + params[3]*t + params[4]*P*t + params[5]*t*t + params[6]*np.sqrt(P) + params[7]*np.log(P)
