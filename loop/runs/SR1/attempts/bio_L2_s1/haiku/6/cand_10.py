import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t - params[2]*P**params[3]/(params[4]**params[3] + P**params[3]) + params[5]*t*P
