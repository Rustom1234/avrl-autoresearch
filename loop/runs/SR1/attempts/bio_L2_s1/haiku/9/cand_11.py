import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P + params[3]*t*P + params[4]*np.exp(-((P-params[5])/params[6])**2) + params[7]*np.exp(-P)
