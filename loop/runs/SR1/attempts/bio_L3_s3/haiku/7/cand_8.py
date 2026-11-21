import numpy as np
def equation(t, P, params):
    g = np.exp(-(t-params[5])**2/params[6])
    return params[0] + params[1]*P + params[2]*P**2 + (params[3] + params[4]*P)*g + params[7]*t*P + params[8]*t
