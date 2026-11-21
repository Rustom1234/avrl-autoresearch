import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P**params[2]*(1+params[3]*t**params[4]) + params[5]*P + params[6]*P**2
