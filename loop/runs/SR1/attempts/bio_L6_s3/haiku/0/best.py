import numpy as np
def equation(t, P, params):
    s = 1-np.exp(-params[7]*t)
    base = params[0]+params[1]*P+params[2]*P**2+params[3]*P**3
    return base + (params[4]+params[5]*P+params[6]*P**2)*s
