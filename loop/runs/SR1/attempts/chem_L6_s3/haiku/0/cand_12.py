import numpy as np
def equation(t, A, params):
    S = params[0]*t**params[1]/(params[2]**params[1]+t**params[1])*np.exp(-params[3]*t)
    return S - params[4]*A**1.5*t**2
