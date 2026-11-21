import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*t + params[2]*t*t) - (params[3] + params[4]*t)*np.abs(P)**params[5]