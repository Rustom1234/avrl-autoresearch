import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P**2 + params[3]*t + params[4]*t**2 + params[5]*t*P + params[6]*t**2*P + params[7]*t*P**2 + params[8]*np.log(t)*P + params[9]*np.log(t)
