import numpy as np
def equation(t, P, params):
    s = 1-np.exp(-params[7]*t)
    return (params[0]+params[1]*P+params[2]*P**2) + (params[3]+params[4]*P+params[5]*P**2)*s + params[6]*s*0
