import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*P + params[2]*P*P + params[3]*t + params[4]*t*t
            + params[5]*P*t + params[6]*P*t*t + params[7]*P*P*t + params[8]*np.sqrt(t) + params[9]*P*np.sqrt(t))
