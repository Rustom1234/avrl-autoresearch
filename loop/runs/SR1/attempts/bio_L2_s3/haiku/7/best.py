import numpy as np
def equation(t, P, params):
    lp = np.log(P)
    return (params[0] + params[1]*t + params[2]*P + params[3]*P*P + params[4]*t*P
            + params[5]*t*t + params[6]*P*lp + params[7]*t*P*lp + params[8]*lp
            + params[9]*t*lp)
