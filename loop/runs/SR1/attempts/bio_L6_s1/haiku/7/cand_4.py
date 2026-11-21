import numpy as np
def equation(t, P, params):
    return params[0]*P + params[1]*P**2 + params[2]*P*t + params[3]*P**2*t + params[4] + params[5]*t + params[6]*np.exp(-params[7]*t)*P + params[8]*np.exp(-params[9]*t)
