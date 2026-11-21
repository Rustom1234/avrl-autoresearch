import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*P + params[2]*t + params[3]*t**2 + params[4]*t**3
            + params[5]*t*P + params[6]*t**2*P + params[7]*np.sqrt(P) + params[8]*np.exp(-t) + params[9]*np.log(t))
