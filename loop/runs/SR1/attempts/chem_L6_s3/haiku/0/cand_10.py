import numpy as np
def equation(t, A, params):
    S = params[0]*(1-np.exp(-(t/params[2])**params[1]))*np.exp(-params[3]*t)
    return S - params[4]*A**params[5]*t**params[6]
