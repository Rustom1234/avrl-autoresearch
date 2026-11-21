import numpy as np
def equation(t, P, params):
    return params[0]*P*(1-P/params[1])*np.exp(-params[2]*t) + params[3]*P*np.sin(params[4]*t)
