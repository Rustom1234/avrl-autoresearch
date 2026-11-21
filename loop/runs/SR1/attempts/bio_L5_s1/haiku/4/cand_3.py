import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*np.exp(-params[3]*P) + params[4]*np.log(P) + params[5]*np.sin(params[6]*t)
