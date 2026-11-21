import numpy as np
def equation(t, P, params):
    K = params[2]
    return (params[0] + params[1]*t) * P * np.log(np.abs(K)/P)
