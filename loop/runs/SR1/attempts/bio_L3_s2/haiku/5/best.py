import numpy as np
def equation(t, P, params):
    w = params[0]
    return (params[1] + params[2]*P + params[3]*P*P + params[4]*t + params[5]*P*t
            + (params[6] + params[7]*P)*np.sin(w*t) + (params[8] + params[9]*P)*np.cos(w*t))
