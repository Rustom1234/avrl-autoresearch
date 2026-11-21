import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*P + params[2]*P*P + params[3]*t + params[4]*t*t
            + params[5]*P*t + params[6]*np.sqrt(t) + params[7]*np.exp(-t) + params[8]*P*np.exp(-t) + params[9]*np.log(t))
