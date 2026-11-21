import numpy as np
def equation(t, P, params):
    return params[0]*P**2 + params[1] + params[2]/(1+params[3]*P)