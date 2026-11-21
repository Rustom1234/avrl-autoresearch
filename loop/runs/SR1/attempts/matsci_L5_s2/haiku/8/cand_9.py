import numpy as np
def equation(eps, T, params):
    return params[0]*((eps+params[1])**params[2]-params[1]**params[2])+params[3]*(1-np.exp(-params[4]*eps))