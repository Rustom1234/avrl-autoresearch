import numpy as np
def equation(t, P, params):
    K = params[1] + params[2]*t
    return (params[0] + params[3]*t)*(1 - np.abs(P/K)**params[4])