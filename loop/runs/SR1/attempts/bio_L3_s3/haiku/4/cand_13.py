import numpy as np
def equation(t, P, params):
    return (params[0]+params[1]*P+params[2]*P**2)*(1+params[3]*np.exp(-params[4]*t)) + params[5]*P*t/(t+params[6]) + params[7]*t/(t+params[8])
