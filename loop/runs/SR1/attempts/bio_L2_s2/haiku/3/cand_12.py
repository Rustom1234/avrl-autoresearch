import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*t)*P*(1 - P/params[2]) + params[3]*np.exp(-params[4]*t) + params[5]*np.sin(params[6]*t) + params[7] + params[8]*t
